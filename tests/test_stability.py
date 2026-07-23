from locium.stability import merge_coords


def test_merge_keeps_existing_coordinates_untouched():
    previous = {"a": [1.0, 2.0], "b": [3.0, 4.0]}
    fresh = {"a": [99.0, 99.0], "b": [88.0, 88.0], "c": [5.0, 6.0]}
    merged = merge_coords(previous, fresh, refit=False)
    assert merged["a"] == [1.0, 2.0]
    assert merged["b"] == [3.0, 4.0]


def test_merge_adds_new_drawers():
    merged = merge_coords({"a": [1.0, 2.0]}, {"a": [9.0, 9.0], "c": [5.0, 6.0]}, refit=False)
    assert merged["c"] == [5.0, 6.0]


def test_merge_drops_deleted_drawers():
    merged = merge_coords({"a": [1.0, 2.0], "gone": [7.0, 7.0]}, {"a": [9.0, 9.0]}, refit=False)
    assert "gone" not in merged


def test_refit_replaces_everything():
    merged = merge_coords({"a": [1.0, 2.0]}, {"a": [9.0, 9.0]}, refit=True)
    assert merged["a"] == [9.0, 9.0]
