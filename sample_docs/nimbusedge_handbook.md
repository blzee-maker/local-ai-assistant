# NimbusEdge X1 — Product & Support Handbook

*Internal reference document. Version 3.2, revised March 2026.*

## Overview

The **NimbusEdge X1** is a compact edge-AI appliance built by Halden Systems for
running small language models fully offline. It is marketed to hospitals, law
firms, and defense contractors where data must never leave the premises.

Halden Systems was founded in **2019** by Priya Venkataraman and Tomas Reinholt
in Tallinn, Estonia. The NimbusEdge product line launched in **August 2024**.

## Hardware specifications

| Component | Specification |
|-----------|---------------|
| Processor | Halden HX-7 octa-core NPU |
| Memory | 24 GB LPDDR5X unified |
| Storage | 1 TB NVMe (encrypted at rest) |
| Max model context | 16,384 tokens |
| Idle power draw | 6 watts |
| Peak power draw | 41 watts |
| Operating temperature | 0°C to 45°C |
| Dimensions | 148mm × 148mm × 34mm |
| Weight | 620 grams |

The X1 ships with three preloaded models: **Aurora-2B**, **Aurora-7B**, and the
code-specialized **Aurora-Coder-3B**. Additional models can be side-loaded over
USB-C only — there is deliberately no network model download path.

## Warranty and support

Every NimbusEdge X1 includes a **26-month limited hardware warranty** from the
date of purchase. Extended coverage ("NimbusCare Plus") adds an additional 24
months and priority replacement.

Support is available exclusively through the offline support portal or by email
at **support@haldensystems.example**. Standard response time is **two business
days**; NimbusCare Plus customers receive a **four-hour** response guarantee.

The device does **not** phone home. All telemetry is stored locally and can be
exported by the administrator as a signed JSON bundle.

## Security posture

- All storage is encrypted at rest using XTS-AES-256.
- The firmware enforces **secure boot** with Halden-signed images only.
- A physical **kill switch** on the rear panel severs all wireless radios.
- Factory reset requires the physical **recovery key**, a 40-character code
  printed on a card included in the box. Halden cannot recover a lost key.

## Compliance

The NimbusEdge X1 is certified for **HIPAA** and **GDPR** deployment and holds a
**FIPS 140-3** cryptographic module validation (certificate #HX7-2025-0114).

For regulated deployments, Halden recommends enabling **Sealed Mode**, which
disables all USB ports and locks the model set. Sealed Mode can only be disabled
on-site with the recovery key.
