def test_hull_has_correct_structure(simplex_hull):
    assert len(simplex_hull.points) == 5
    assert len(simplex_hull.simplices) > 0