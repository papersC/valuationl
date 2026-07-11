// Chaincode-as-a-Service build of the valcert contract. Identical contract
// logic to valcert.go, but with v2 imports and a shim.ChaincodeServer main so
// the peer connects to this process over gRPC (avoids the peer-driven docker
// image build, which is incompatible with very recent Docker engines). This is
// the file compiled into the CCAAS container; valcert.go documents the contract
// for readers and is not built here.
//go:build ccaas

package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/hyperledger/fabric-chaincode-go/v2/shim"
	"github.com/hyperledger/fabric-contract-api-go/v2/contractapi"
)

type SmartContract struct {
	contractapi.Contract
}

type Certificate struct {
	CertID      string `json:"certId"`
	Digest      string `json:"digest"`
	CommittedTx string `json:"committedTx"`
}

const evidencePrefix = "EV_"
const certPrefix = "CERT_"

func isHex32(s string) bool {
	if len(s) != 64 {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}

func (s *SmartContract) AnchorEvidence(ctx contractapi.TransactionContextInterface, eid string, digest string) error {
	if !isHex32(digest) {
		return fmt.Errorf("evidence digest must be a 32-byte (64-hex) SHA-256 hash")
	}
	exists, err := ctx.GetStub().GetState(evidencePrefix + eid)
	if err != nil {
		return err
	}
	if exists != nil {
		return fmt.Errorf("evidence %s already anchored (append-only: no rewrite)", eid)
	}
	return ctx.GetStub().PutState(evidencePrefix+eid, []byte(digest))
}

func (s *SmartContract) CommitCertificate(ctx contractapi.TransactionContextInterface, certID string, digest string) error {
	if !isHex32(digest) {
		return fmt.Errorf("certificate digest must be a 32-byte (64-hex) SHA-256 hash")
	}
	key := certPrefix + certID
	exists, err := ctx.GetStub().GetState(key)
	if err != nil {
		return err
	}
	if exists != nil {
		return fmt.Errorf("certificate %s already committed (append-only: no rewrite)", certID)
	}
	cert := Certificate{CertID: certID, Digest: digest, CommittedTx: ctx.GetStub().GetTxID()}
	b, err := json.Marshal(cert)
	if err != nil {
		return err
	}
	return ctx.GetStub().PutState(key, b)
}

func (s *SmartContract) GetCertificate(ctx contractapi.TransactionContextInterface, certID string) (*Certificate, error) {
	b, err := ctx.GetStub().GetState(certPrefix + certID)
	if err != nil {
		return nil, err
	}
	if b == nil {
		return nil, fmt.Errorf("certificate %s not found", certID)
	}
	var cert Certificate
	if err := json.Unmarshal(b, &cert); err != nil {
		return nil, err
	}
	return &cert, nil
}

func (s *SmartContract) VerifyCertificate(ctx contractapi.TransactionContextInterface, certID string, recomputedDigest string) (bool, error) {
	cert, err := s.GetCertificate(ctx, certID)
	if err != nil {
		return false, err
	}
	return cert.Digest == recomputedDigest, nil
}

func (s *SmartContract) ResolveEvidence(ctx contractapi.TransactionContextInterface, eid string) (string, error) {
	b, err := ctx.GetStub().GetState(evidencePrefix + eid)
	if err != nil {
		return "", err
	}
	if b == nil {
		return "", fmt.Errorf("evidence %s not anchored", eid)
	}
	return string(b), nil
}

// Digest is a deterministic helper used by clients to confirm they compute
// commitments the same way the ledger does.
func Digest(obj interface{}) string {
	b, _ := json.Marshal(obj)
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

type serverConfig struct {
	CCID    string
	Address string
}

func main() {
	config := serverConfig{
		CCID:    os.Getenv("CHAINCODE_ID"),
		Address: os.Getenv("CHAINCODE_SERVER_ADDRESS"),
	}
	cc, err := contractapi.NewChaincode(&SmartContract{})
	if err != nil {
		log.Panicf("error creating valcert chaincode: %s", err)
	}
	server := &shim.ChaincodeServer{
		CCID:     config.CCID,
		Address:  config.Address,
		CC:       cc,
		TLSProps: getTLSProperties(),
	}
	if err := server.Start(); err != nil {
		log.Panicf("error starting valcert chaincode: %s", err)
	}
}

func getTLSProperties() shim.TLSProperties {
	tlsDisabledStr := getEnvOrDefault("CHAINCODE_TLS_DISABLED", "true")
	key := getEnvOrDefault("CHAINCODE_TLS_KEY", "")
	cert := getEnvOrDefault("CHAINCODE_TLS_CERT", "")
	clientCACert := getEnvOrDefault("CHAINCODE_CLIENT_CA_CERT", "")
	tlsDisabled := getBoolOrDefault(tlsDisabledStr, false)
	var keyBytes, certBytes, clientCACertBytes []byte
	var err error
	if !tlsDisabled {
		keyBytes, err = os.ReadFile(key)
		if err != nil {
			log.Panicf("error while reading the crypto file: %s", err)
		}
		certBytes, err = os.ReadFile(cert)
		if err != nil {
			log.Panicf("error while reading the crypto file: %s", err)
		}
	}
	if clientCACert != "" {
		clientCACertBytes, err = os.ReadFile(clientCACert)
		if err != nil {
			log.Panicf("error while reading the crypto file: %s", err)
		}
	}
	return shim.TLSProperties{
		Disabled:      tlsDisabled,
		Key:           keyBytes,
		Cert:          certBytes,
		ClientCACerts: clientCACertBytes,
	}
}

func getEnvOrDefault(env, defaultVal string) string {
	value, ok := os.LookupEnv(env)
	if !ok {
		value = defaultVal
	}
	return value
}

func getBoolOrDefault(value string, defaultVal bool) bool {
	parsed, err := strconv.ParseBool(value)
	if err != nil {
		return defaultVal
	}
	return parsed
}
