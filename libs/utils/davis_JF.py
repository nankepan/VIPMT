import numpy as np


def db_eval_boundary(foreground_mask,gt_mask,bound_th=0.008):
	assert np.atleast_3d(foreground_mask).shape[2] == 1

	bound_pix = bound_th if bound_th >= 1 else \
			np.ceil(bound_th*np.linalg.norm(foreground_mask.shape))

	# Get the pixel boundaries of both masks
	fg_boundary = seg2bmap(foreground_mask);
	gt_boundary = seg2bmap(gt_mask);

	from skimage.morphology import binary_dilation,disk

	fg_dil = binary_dilation(fg_boundary,disk(bound_pix))
	gt_dil = binary_dilation(gt_boundary,disk(bound_pix))

	# Get the intersection
	gt_match = gt_boundary * fg_dil
	fg_match = fg_boundary * gt_dil

	# Area of the intersection
	n_fg     = np.sum(fg_boundary)
	n_gt     = np.sum(gt_boundary)

	#% Compute precision and recall
	if n_fg == 0 and  n_gt > 0:
		precision = 1
		recall = 0
	elif n_fg > 0 and n_gt == 0:
		precision = 0
		recall = 1
	elif n_fg == 0  and n_gt == 0:
		precision = 1
		recall = 1
	else:
		precision = np.sum(fg_match)/float(n_fg)
		recall    = np.sum(gt_match)/float(n_gt)

	# Compute F measure
	if precision + recall == 0:
		F = 0
	else:
		F = 2*precision*recall/(precision+recall);

	return F

def seg2bmap(seg,width=None,height=None):
	seg = seg.astype(np.bool)
	seg[seg>0] = 1

	assert np.atleast_3d(seg).shape[2] == 1

	width  = seg.shape[1] if width  is None else width
	height = seg.shape[0] if height is None else height

	h,w = seg.shape[:2]

	ar1 = float(width) / float(height)
	ar2 = float(w) / float(h)

	assert not (width>w | height>h | abs(ar1-ar2)>0.01),\
			'Can''t convert %dx%d seg to %dx%d bmap.'%(w,h,width,height)

	e  = np.zeros_like(seg)
	s  = np.zeros_like(seg)
	se = np.zeros_like(seg)

	e[:,:-1]    = seg[:,1:]
	s[:-1,:]    = seg[1:,:]
	se[:-1,:-1] = seg[1:,1:]

	b        = seg^e | seg^s | seg^se
	b[-1,:]  = seg[-1,:]^e[-1,:]
	b[:,-1]  = seg[:,-1]^s[:,-1]
	b[-1,-1] = 0

	if w == width and h == height:
		bmap = b
	else:
		bmap = np.zeros((height,width))
		for x in range(w):
			for y in range(h):
				if b[y,x]:
					j = 1+floor((y-1)+height / h)
					i = 1+floor((x-1)+width  / h)
					bmap[j,i] = 1;

	return bmap

def db_eval_iou(annotation,segmentation):
    annotation   = annotation.astype(np.bool)
    segmentation = segmentation.astype(np.bool)

    if np.isclose(np.sum(annotation),0) and np.isclose(np.sum(segmentation),0):
        return 1
    else:
        return np.sum((annotation & segmentation)) / \
                np.sum((annotation | segmentation),dtype=np.float32)
