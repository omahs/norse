from norse.torch.functional import receptive_field


def test_zero_derivative():
    f = receptive_field.spatial_receptive_fields_with_derivatives(2, 2, 2, 9)
    assert f.shape == (8, 9, 9)


def test_first_derivative_number():
    f = receptive_field.spatial_receptive_fields_with_derivatives(2, 2, 2, 9, 1)
    assert f.shape == (8 * 4, 9, 9)


def test_first_derivative():
    f = receptive_field.spatial_receptive_fields_with_derivatives(2, 2, 2, 9, [(1, 1)])
    assert f.shape == (8, 9, 9)


def test_several_derivatives():
    f = receptive_field.spatial_receptive_fields_with_derivatives(
        2, 2, 2, 9, [(1, 1), (3, 2)]
    )
    assert f.shape == (16, 9, 9)


def test_10th_derivative():
    f = receptive_field.spatial_receptive_fields_with_derivatives(2, 2, 2, 9, [(3, 10)])
    assert f.shape == (8, 9, 9)
