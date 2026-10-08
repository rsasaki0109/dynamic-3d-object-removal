import pytest
from scripts.validate_unseen_av2_neighbors import require_unseen


def test_previously_examined_log_is_rejected_before_evaluation():
    with pytest.raises(ValueError, match='previously examined'):
        require_unseen('old-log', {'old-log', 'other-log'})
    require_unseen('new-log', {'old-log', 'other-log'})
