"""CLI: python -m hs_rag_classifier.finetune {build,check-equivalence} (see build_set),
{train,tf-score} (see train); val scoring and the gate: python -m hs_rag_classifier.finetune.score_val."""

import sys

if len(sys.argv) > 1 and sys.argv[1] in ("train", "tf-score"):
    from hs_rag_classifier.finetune.train import main
else:
    from hs_rag_classifier.finetune.build_set import main

sys.exit(main())
