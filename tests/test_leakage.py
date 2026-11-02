"""Leakage-invariance test (Section 2.4): replacing the ΣPAHs values and optical descriptors of
held-out samples with random values must leave their predictions unchanged."""
from cascadepahs.validation import folds_blocks, leakage_invariance


def test_heldout_predictions_do_not_depend_on_heldout_labels_or_optics(dasha, small_solace):
    folds = folds_blocks(dasha, 4)
    folds = [folds[0], folds[3], folds[-1]]            # two summer blocks and the winter block
    worst = leakage_invariance(lambda: small_solace(), dasha, folds, seed=3)
    assert worst < 1e-10
