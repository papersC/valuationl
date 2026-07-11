// Package main implements the valuation-certificate chaincode (chaincode name
// "valcert") for the audit-ready valuation pipeline.
//
// The paper's assurance is that the on-chain footprint of an appraisal is a
// single constant-size record: a 32-byte SHA-256 commitment that binds the
// evidence-set digest, the pinned model/index versions, the value and interval,
// and the narrative digest. This chaincode enforces exactly that. It never
// stores an off-chain payload; only fixed-width digests are written to the
// world state, so the ledger cost is O(1) per valuation regardless of how many
// comparables or narrative claims the appraisal used.
package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"

	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// SmartContract provides functions for anchoring evidence and committing and
// verifying valuation certificates.
type SmartContract struct {
	contractapi.Contract
}

// Certificate is the constant-size on-chain record. Digest is the 64-hex-char
// (32-byte) SHA-256 commitment binding all appraisal fields, computed off-chain
// and re-verifiable by any peer. CommittedTx pins the transaction that wrote it.
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

// AnchorEvidence records the digest of one evidence item (a comparable sale,
// lease, index point, or attribute snapshot). The payload lives off-chain in a
// content-addressed store; only its 32-byte digest is anchored here.
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

// CommitCertificate writes one constant-size certificate. digest is the SHA-256
// commitment over the appraisal's bound fields, computed off-chain. This is the
// only per-appraisal write, and its size is independent of the evidence volume.
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

// GetCertificate returns the committed certificate record for replay.
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

// VerifyCertificate re-checks a locally recomputed digest against the committed
// one. A true result is the replayability guarantee: refetching the pinned
// evidence and recomputing the appraisal reproduces the on-chain commitment.
func (s *SmartContract) VerifyCertificate(ctx contractapi.TransactionContextInterface, certID string, recomputedDigest string) (bool, error) {
	cert, err := s.GetCertificate(ctx, certID)
	if err != nil {
		return false, err
	}
	return cert.Digest == recomputedDigest, nil
}

// ResolveEvidence returns the anchored digest for an evidence id, or an error if
// it was never anchored (used by the auditor's anchoring clause).
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

// Digest is a deterministic helper: the SHA-256 over a canonical JSON object.
// Provided so a client can confirm it computes commitments the same way.
func Digest(obj interface{}) string {
	b, _ := json.Marshal(obj)
	h := sha256.Sum256(b)
	return hex.EncodeToString(h[:])
}

func main() {
	cc, err := contractapi.NewChaincode(&SmartContract{})
	if err != nil {
		panic(fmt.Sprintf("error creating valcert chaincode: %v", err))
	}
	if err := cc.Start(); err != nil {
		panic(fmt.Sprintf("error starting valcert chaincode: %v", err))
	}
}
