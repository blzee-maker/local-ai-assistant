# Local RAG retrieval benchmark

LLM generation is excluded. Warm statistics include only the timed retrieval calls; the first cold retrieval and five warm-up calls per top-k are excluded from them.

## Measured latency

| top_k | stage | median (ms) | mean (ms) | p95 (ms) | min (ms) | max (ms) |
|---:|---|---:|---:|---:|---:|---:|
| 4 | embedding | 29.89074999982222 | 31.584846666641575 | 46.715600000197796 | 19.84179999999469 | 61.58259999983784 |
| 4 | search | 0.06304999988060445 | 0.08022666662428593 | 0.24769500025740826 | 0.04379999973025406 | 0.35970000044471817 |
| 4 | total | 29.94884999975511 | 31.67848888891361 | 46.93322499988425 | 19.9176000000989 | 61.67219999997542 |
| 5 | embedding | 30.568749999929423 | 30.820551111087802 | 33.79310499994972 | 19.173700000010285 | 110.67110000021785 |
| 5 | search | 0.0629000001026725 | 0.07103444445925157 | 0.11932499996873946 | 0.048100000185513636 | 0.3011000003425579 |
| 5 | total | 30.64324999991186 | 30.90255444442644 | 33.910950000085904 | 19.24060000010286 | 110.74830000006841 |

## Run and corpus

- Queries: 30; measured runs per query: 3; measured retrievals per top_k: 90.
- Cold first retrieval: 395.8112999998775 ms (RagService construction (including its embedding dimension probe and FAISS index load) plus the first retrieval; excludes Python interpreter startup and imports.)
- Cold-start caveat: The model was already present in the local cache and downloads were disabled. The corpus index was built immediately before measurement, so filesystem-cache state was not controlled.
- Corpus: 1 files, 6 chunks/vectors, dimension 384, IndexFlatIP.
- Corpus files: nimbusedge_handbook.md.
- Default top_k: 4.
- Cold first-query embedding: 34.29099999993923 ms; FAISS search: 0.1612000000932312 ms.

## Environment

- CPU: 11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz.
- RAM: 8358518784 bytes total; 1072209920 bytes available before run.
- OS: Windows-11-10.0.26200-SP0; Python: 3.13.1.
- Libraries: faiss 1.15.0, onnxruntime 1.28.0, numpy 2.5.1, fastembed 0.8.0.
- ONNX Runtime providers available: AzureExecutionProvider, CPUExecutionProvider; selected: CPUExecutionProvider.
- GPU used: False.
- Other processes with at least 200 MiB RSS: MemCompression (453.328125 MiB), Code.exe (428.1796875 MiB), Code.exe (272.890625 MiB), Code.exe (247.68359375 MiB), copilot-runtime.exe (229.625 MiB), MsMpEng.exe (219.328125 MiB).

## Caveats

- The in-repository sample corpus contains 1 file and 6 vectors; these measurements do not establish latency at larger corpus sizes.
- The cold-start figure measures a fresh benchmark process with local model files already cached; it excludes interpreter startup/imports and any first-time model download.
- Other background processes were active; their observed memory use is listed in the environment section.

## Retrieval sanity checks

- Query: What is the peak power draw of the X1?
  - Top result: Hardware specifications
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
  - Source: nimbusedge_handbook.md; score: 0.3671985864639282; checked for `41 watts`.
- Query: Who founded Halden Systems, and in what year?
  - Top result: Overview
The **NimbusEdge X1** is a compact edge-AI appliance built by Halden Systems for
running small language models fully offline. It is marketed to hospitals, law
firms, and defense contractors where data must never leave the premises.

Halden Systems was founded in **2019** by Priya Venkataraman and Tomas Reinholt
in Tallinn, Estonia. The NimbusEdge product line launched in **August 2024**.
  - Source: nimbusedge_handbook.md; score: 0.48719245195388794; checked for `Priya Venkataraman`.
- Query: How can the wireless radios be physically disabled?
  - Top result: Security posture
- All storage is encrypted at rest using XTS-AES-256.
- The firmware enforces **secure boot** with Halden-signed images only.
- A physical **kill switch** on the rear panel severs all wireless radios.
- Factory reset requires the physical **recovery key**, a 40-character code
  printed on a card included in the box. Halden cannot recover a lost key.
  - Source: nimbusedge_handbook.md; score: 0.43870508670806885; checked for `kill switch`.
