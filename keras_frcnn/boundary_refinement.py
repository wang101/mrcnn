def boundary_refinement(bbox):
    assert len(bbox.shape)==2
    threshold = 0.5
    valid_box = 0
    FILL_IN = True

    row,col = bbox.shape
    boolbox = np.where(bbox!=0,1,0)
    n_pixels = np.sum(np.sum(boolbox))
    if n_pixels*1.0/(row*col) > threshold:
        valid_box = 1

    if FILL_IN:

