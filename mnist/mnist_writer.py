"""
Writes the predictions to a csv in Kaggle submission format
"""

def write_predictions(preds, filename):
    with open(filename, 'w') as writer:
        writer.write('"ImageId","Label"\n')
        count = 0
        for p in preds:
            count += 1
            writer.write(str(count) + ',"' + str(p) + '"\n')