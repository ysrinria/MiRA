

def class_label_map(str_label):
    if str_label == False:
        numeric_label = -1
    else:
        numeric_label = -1
    return numeric_label


def DFEW_class_label_map(str_label):
    if str_label == 'Disgust':
        numeric_label = 0
    elif str_label == 'Sad':
        numeric_label = 1
    elif str_label == 'Neutral':
        numeric_label = 2
    elif str_label == 'Surprise':
        numeric_label = 3
    elif str_label == 'Angry':
        numeric_label = 4
    elif str_label == 'Happy':
        numeric_label = 5
    elif str_label == 'Fear':
        numeric_label = 6
    else:
        numeric_label = -1
    return numeric_label


def MAFW_class_label_map(str_label):
    if str_label == 'Contempt':
        numeric_label = 0
    elif str_label == 'Anxiety':
        numeric_label = 1
    elif str_label == 'Neutral':
        numeric_label = 2
    elif str_label == 'Sadness':
        numeric_label = 3
    elif str_label == 'Anger':
        numeric_label = 4
    elif str_label == 'Disgust':
        numeric_label = 5
    elif str_label == 'Fear':
        numeric_label = 6
    elif str_label == 'Surprise':
        numeric_label = 7
    elif str_label == 'Happiness':
        numeric_label = 8
    elif str_label == 'Helplessness':
        numeric_label = 9
    elif str_label == 'Disappointment':
        numeric_label = 10
    else:
        numeric_label = -1
    return numeric_label


def AVCAFFE_V_class_label_map(str_label_V):
    if str_label_V == 'Unpleasant':
        numeric_label_V = 0
    elif str_label_V == 'Unsatisfied':
        numeric_label_V = 1
    elif str_label_V == 'Neutral':
        numeric_label_V = 2
    elif str_label_V == 'Pleased':
        numeric_label_V = 3
    elif str_label_V == 'Pleasant':
        numeric_label_V = 4
    else:
        numeric_label_V = -1
    return numeric_label_V


def AVCAFFE_A_class_label_map(str_label_A):
    if str_label_A == 'Excited':
        numeric_label_A = 0
    elif str_label_A == 'Neutral':
        numeric_label_A = 1
    elif str_label_A == 'Dull':
        numeric_label_A = 2
    elif str_label_A == 'Wide-awake':
        numeric_label_A = 3
    elif str_label_A == 'Calm':
        numeric_label_A = 4
    else:
        numeric_label_A = -1
    return numeric_label_A


def FERV39k_class_label_map(str_label):
    if str_label == 'Disgust':
        numeric_label = 0
    elif str_label == 'Sad':
        numeric_label = 1
    elif str_label == 'Neutral':
        numeric_label = 2
    elif str_label == 'Surprise':
        numeric_label = 3
    elif str_label == 'Angry':
        numeric_label = 4
    elif str_label == 'Happy':
        numeric_label = 5
    elif str_label == 'Fear':
        numeric_label = 6
    else:
        numeric_label = -1
    return numeric_label