def get_sky_mask(pred, index=2):
    # index 2 is sky
    if index is not None:
        pred = pred.copy()
        pred[pred != index] = -1
        # print(f'{names[index + 1]}:')
    return pred


