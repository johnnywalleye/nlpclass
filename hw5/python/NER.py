import sys, os
from subprocess import Popen, PIPE
from FeatureFactory import FeatureFactory

"""
Do not modify this class
The submit script does not use this class
It directly calls the methods of FeatureFactory and MEMM classes.
"""
def main(argv):
    # defaults
    if len(argv) == 0:
        argv.append("../data/train")
        argv.append("../data/dev")
    elif len(argv) < 2:
        print 'USAGE: python NER.py trainFile testFile'
        exit(0)

    # Set this to -print to print
    printOp = ''
    if len(argv) > 2:
        printOp = '-print'

    excluded = set()

    featureFactory = FeatureFactory(excluded)

    # read the train and test data
    trainData = featureFactory.readData(argv[0])
    testData = featureFactory.readData(argv[1])

    # add the features
    trainDataWithFeatures = featureFactory.setFeaturesTrain(trainData);
    testDataWithFeatures = featureFactory.setFeaturesTest(testData);

    # write the updated data into JSON files
    featureFactory.writeData(trainDataWithFeatures, 'trainWithFeatures');
    featureFactory.writeData(testDataWithFeatures, 'testWithFeatures');

    # run MEMM
    output = Popen(['java','-cp', 'classes', '-Xmx4G' ,'MEMM'
                    ,'trainWithFeatures.json', 'testWithFeatures.json',
                    printOp], stdout=PIPE).communicate()[0]

    print output

if __name__ == '__main__':
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        main([])



