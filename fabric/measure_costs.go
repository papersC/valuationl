// measure_costs.go — drive the valcert chaincode from a Fabric Gateway client
// and measure the on-chain costs the paper claims:
//
//   1. Commit latency: end-to-end SubmitTransaction (submit -> 2-org endorse ->
//      order -> commit) per real Dubai certificate. Mean/median/p95/p99.
//   2. Throughput: sequential and concurrent (goroutine pool) tx/s.
//   3. O(1) footprint: certificates whose evidence set ranges 4..64 comparables
//      all commit a constant 64-hex (32-byte) digest; the committed record size
//      is independent of evidence volume.
//   4. Integrity: VerifyCertificate(correct)=true, (tampered)=false;
//      AnchorEvidence rejects a second write to the same id (append-only).
//
// One gRPC connection is reused for all transactions so the reported latency is
// the network's, not per-call client startup.
package main

import (
	"bytes"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"sync/atomic"
	"time"

	"github.com/hyperledger/fabric-gateway/pkg/client"
	"github.com/hyperledger/fabric-gateway/pkg/identity"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
)

const (
	mspID        = "Org1MSP"
	channelName  = "valuation"
	chaincodeNm  = "valcert"
	peerEndpoint = "localhost:17051"
	gatewayPeer  = "peer0.org1.valcert.io"
)

var cryptoBase = "/root/fabric/fabric-samples/tn-iso/organizations/peerOrganizations/org1.valcert.io"

type Cert struct {
	CertID          string            `json:"certId"`
	CertDigest      string            `json:"certDigest"`
	NEvidence       int               `json:"nEvidence"`
	EvidenceDigests map[string]string `json:"evidenceDigests"`
	Value           float64           `json:"value"`
	K               int               `json:"k"`
}

type CertsFile struct {
	TestYear     int    `json:"test_year"`
	NLatency     int    `json:"n_latency"`
	LatencyCerts []Cert `json:"latency_certs"`
	SweepCerts   []Cert `json:"sweep_certs"`
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}

func firstFile(dir string) string {
	entries, err := os.ReadDir(dir)
	must(err)
	return filepath.Join(dir, entries[0].Name())
}

func newConnection() *grpc.ClientConn {
	tlsCertPath := cryptoBase + "/peers/peer0.org1.valcert.io/tls/ca.crt"
	pem, err := os.ReadFile(tlsCertPath)
	must(err)
	pool := x509.NewCertPool()
	pool.AppendCertsFromPEM(pem)
	tc := credentials.NewClientTLSFromCert(pool, gatewayPeer)
	conn, err := grpc.NewClient(peerEndpoint, grpc.WithTransportCredentials(tc))
	must(err)
	return conn
}

func newIdentity() *identity.X509Identity {
	certPath := firstFile(cryptoBase + "/users/User1@org1.valcert.io/msp/signcerts")
	pem, err := os.ReadFile(certPath)
	must(err)
	cert, err := identity.CertificateFromPEM(pem)
	must(err)
	id, err := identity.NewX509Identity(mspID, cert)
	must(err)
	return id
}

func newSign() identity.Sign {
	keyPath := firstFile(cryptoBase + "/users/User1@org1.valcert.io/msp/keystore")
	pem, err := os.ReadFile(keyPath)
	must(err)
	key, err := identity.PrivateKeyFromPEM(pem)
	must(err)
	sign, err := identity.NewPrivateKeySign(key)
	must(err)
	return sign
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	idx := int(p * float64(len(sorted)-1))
	return sorted[idx]
}

func main() {
	runTag := os.Args[1] // unique prefix so cert ids don't collide across runs
	raw, err := os.ReadFile("certs.json")
	must(err)
	var cf CertsFile
	must(json.Unmarshal(raw, &cf))

	conn := newConnection()
	defer conn.Close()
	gw, err := client.Connect(newIdentity(), client.WithSign(newSign()),
		client.WithClientConnection(conn),
		client.WithEvaluateTimeout(30*time.Second),
		client.WithEndorseTimeout(30*time.Second),
		client.WithSubmitTimeout(30*time.Second),
		client.WithCommitStatusTimeout(1*time.Minute))
	must(err)
	defer gw.Close()
	contract := gw.GetNetwork(channelName).GetContract(chaincodeNm)

	results := map[string]interface{}{
		"channel": channelName, "chaincode": chaincodeNm,
		"endorsement": "majority (Org1MSP AND Org2MSP)",
		"peer_endpoint": peerEndpoint, "run_tag": runTag,
	}

	// ---- 1+2. commit latency & sequential throughput on real certs ----
	commit := func(id, digest string) (time.Duration, error) {
		t0 := time.Now()
		_, err := contract.SubmitTransaction("CommitCertificate", id, digest)
		return time.Since(t0), err
	}
	// warm-up (JIT, connection priming) — 5 tx, not measured
	for i := 0; i < 5; i++ {
		_, _ = commit(fmt.Sprintf("%s-warm-%d", runTag, i), cf.LatencyCerts[i].CertDigest)
	}
	lat := make([]float64, 0, len(cf.LatencyCerts))
	seqStart := time.Now()
	for i, c := range cf.LatencyCerts {
		d, err := commit(fmt.Sprintf("%s-L-%d-%s", runTag, i, c.CertID), c.CertDigest)
		if err != nil {
			fmt.Printf("commit err %v\n", err)
			continue
		}
		lat = append(lat, float64(d.Microseconds())/1000.0) // ms
	}
	seqWall := time.Since(seqStart).Seconds()
	sort.Float64s(lat)
	sum := 0.0
	for _, v := range lat {
		sum += v
	}
	results["commit_latency_ms"] = map[string]interface{}{
		"n": len(lat), "mean": sum / float64(len(lat)),
		"p50": percentile(lat, 0.50), "p95": percentile(lat, 0.95),
		"p99": percentile(lat, 0.99), "min": lat[0], "max": lat[len(lat)-1],
	}
	results["throughput_sequential_tps"] = float64(len(lat)) / seqWall

	// ---- concurrent throughput ----
	concN := 200
	if concN > len(cf.LatencyCerts) {
		concN = len(cf.LatencyCerts)
	}
	workers := 16
	var done int64
	jobs := make(chan int, concN)
	var wg sync.WaitGroup
	cStart := time.Now()
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := range jobs {
				_, err := commit(fmt.Sprintf("%s-C-%d", runTag, j), cf.LatencyCerts[j].CertDigest)
				if err == nil {
					atomic.AddInt64(&done, 1)
				}
			}
		}()
	}
	for j := 0; j < concN; j++ {
		jobs <- j
	}
	close(jobs)
	wg.Wait()
	cWall := time.Since(cStart).Seconds()
	results["throughput_concurrent_tps"] = float64(done) / cWall
	results["throughput_concurrent_workers"] = workers

	// ---- 3. O(1) footprint across evidence volume ----
	type footPoint struct {
		K, NEvidence, DigestLen, OnchainBytes int
	}
	foot := []footPoint{}
	for i, c := range cf.SweepCerts {
		id := fmt.Sprintf("%s-S-%d-%s", runTag, i, c.CertID)
		_, err := contract.SubmitTransaction("CommitCertificate", id, c.CertDigest)
		if err != nil {
			fmt.Printf("sweep commit err %v\n", err)
			continue
		}
		got, err := contract.EvaluateTransaction("GetCertificate", id)
		must(err)
		foot = append(foot, footPoint{c.K, c.NEvidence, len(c.CertDigest), len(got)})
	}
	results["footprint"] = foot

	// ---- 4. integrity properties ----
	integ := map[string]interface{}{}
	c0 := cf.LatencyCerts[0]
	idv := runTag + "-verify"
	_, err = contract.SubmitTransaction("CommitCertificate", idv, c0.CertDigest)
	must(err)
	okGood, err := contract.EvaluateTransaction("VerifyCertificate", idv, c0.CertDigest)
	must(err)
	integ["verify_correct_digest"] = string(okGood) // "true"
	tampered := "deadbeef" + c0.CertDigest[8:]
	okBad, err := contract.EvaluateTransaction("VerifyCertificate", idv, tampered)
	must(err)
	integ["verify_tampered_digest"] = string(okBad) // "false"
	// append-only: second anchor of same evidence id must error
	eid := runTag + "-ev0"
	someDigest := c0.CertDigest
	_, err = contract.SubmitTransaction("AnchorEvidence", eid, someDigest)
	must(err)
	_, err2 := contract.SubmitTransaction("AnchorEvidence", eid, someDigest)
	integ["reanchor_rejected"] = (err2 != nil)
	results["integrity"] = integ

	out, _ := json.MarshalIndent(results, "", "  ")
	var pretty bytes.Buffer
	json.Indent(&pretty, out, "", "  ")
	must(os.WriteFile("fabric_results.json", out, 0644))
	fmt.Println(string(out))
}
