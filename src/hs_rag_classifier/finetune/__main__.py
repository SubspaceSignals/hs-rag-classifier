"""CLI: python -m hs_rag_classifier.finetune {build,check-equivalence} (see build_set)."""

import sys

from hs_rag_classifier.finetune.build_set import main

sys.exit(main())
