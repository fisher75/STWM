# STWM Semantic-Only TUSB Unfreeze V1 Boundary Audit

Dry audit passed without training steps.

- Boundary ok: `True`
- Total trainable params: `1649543`
- Future semantic state head trainable params: `758660`
- Semantic fusion projection trainable params: `296064`
- TUSB factorized semantic trainable params: `148993`
- TUSB broadcast semantic trainable params: `443520`
- TUSB handshake semantic trainable params: `0`
- TUSB dynamic trainable params: `0`
- Mixed trainable params: `0`
- Stage1 trainable params: `0`
- Trace backbone trainable: `False`

This is the first boundary that actually lets semantic prototype loss update TUSB semantic state while keeping dynamic trace state frozen.
