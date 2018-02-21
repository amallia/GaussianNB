#!/usr/bin/env python
import argparse
import pandas
import numpy
import math
import logging

LOG_FILENAME = "output.log"

CV_NUM = 5

ATTR_NAMES = ["RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe"]
FIELD_NAMES = ["Num"] + ATTR_NAMES + ["Class"]


class GNB_classifier(object):

    def __init__(self, training_set, test_set):
        self.__training_set = training_set
        self.__test_set = test_set
        self.__n = len(self.__training_set)
        self.__prior()
        self.__calculate_mean_variance()

    def __prior(self):
        counts = self.__training_set["Class"].value_counts().to_dict()
        self.__priors = {(k, v / self.__n) for k, v in counts.items()}

    def __calculate_mean_variance(self):
        self.__mean_variance = {}
        for c in self.__training_set["Class"].unique():
            filtered_set = self.__training_set[
                (self.__training_set['Class'] == c)]
            m_v = {}
            for attr_name in ATTR_NAMES:
                m_v[attr_name] = []
                m_v[attr_name].append(filtered_set[attr_name].mean())
                m_v[attr_name].append(
                    math.pow(filtered_set[attr_name].std(), 2))
            self.__mean_variance[c] = m_v

    @staticmethod
    def __calculate_probability(x, mean, variance):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * variance)))
        return (1 / (math.sqrt(2 * math.pi * variance))) * exponent

    def predict(self):
        predictions = {}
        for _, row in self.__test_set.iterrows():
            results = {}
            for k, v in self.__priors:
                p = 0
                for attr_name in ATTR_NAMES:
                    prob = self.__calculate_probability(row[attr_name], self.__mean_variance[
                        k][attr_name][0], self.__mean_variance[k][attr_name][1])
                    if prob > 0:
                        p += math.log(prob)
                results[k] = math.log(v) + p
            predictions[int(row["Num"])] = max([key for key in results.keys() if results[
                key] == results[max(results, key=results.get)]])
        return predictions

    def print_info(self):
        logging.info("Priors for each class: %s", self.__priors)
        logging.info(
            "Means and variance for each class: %s", self.__mean_variance
        )


class zero_r_classifier(object):
    def __init__(self, training_set, test_set):
        self.__test_set = test_set
        classes = training_set["Class"].value_counts().to_dict()
        self.__most_freq_class = max(classes, key=classes.get)

    def predict(self):
        predictions = {}
        for _, row in self.__test_set.iterrows():
            predictions[int(row["Num"])] = self.__most_freq_class
        return predictions


def calculate_accuracy(test_set, predictions):
    correct = 0
    for _, t in test_set.iterrows():
        if t["Class"] == predictions[t["Num"]]:
            correct += 1
    return (correct / len(test_set)) * 100.0


def split_data(data, blocks_num=1, test_block=0):
    blocks = numpy.array_split(data, blocks_num)
    test_set = blocks[test_block]
    if blocks_num > 1:
        del blocks[test_block]
    training_set = pandas.concat(blocks)
    return training_set, test_set


def main():
    logging.basicConfig(filename=LOG_FILENAME, level=logging.INFO)
    parser = argparse.ArgumentParser(
        description='Gaussian Naive Bayes classifier.')
    parser.add_argument('filename', metavar="FILENAME",
                        help='Dataset filename.')
    args = parser.parse_args()
    data = pandas.read_csv(args.filename, names=FIELD_NAMES)
    logging.info("================================================"
                 "================================================")
    logging.info(
        "Running Gaussian Naive Bayes classifier "
        "using the entire dataset for both training and testing.")
    logging.info("================================================"
                 "================================================")
    training_set, test_set = split_data(data)
    classifier = GNB_classifier(training_set, test_set)
    classifier.print_info()
    predictions = classifier.predict()
    logging.info(
        "Predictions in the form (number, predicted class): %s", predictions)
    accuracy = calculate_accuracy(test_set, predictions)
    logging.info("Accuracy for the current experiment: %s", accuracy)
    logging.info("================================================"
                 "================================================\n")

    logging.info("================================================"
                 "================================================")
    logging.info(
        "Running Gaussian Naive Bayes classifier "
        "using %d-fold cross-validation.", CV_NUM)
    logging.info("================================================"
                 "================================================")
    total_accuracy = 0
    for i in range(CV_NUM):
        training_set, test_set = split_data(data, CV_NUM, i)
        classifier = GNB_classifier(training_set, test_set)
        classifier.print_info()
        predictions = classifier.predict()
        logging.info(
            "Predictions in the form (number, predicted class): %s", predictions)
        accuracy = calculate_accuracy(test_set, predictions)
        logging.info("Accuracy for the current experiment: %s", accuracy)
        total_accuracy += accuracy
    logging.info("Total accuracy: %s", total_accuracy / CV_NUM)
    logging.info("================================================"
                 "================================================\n")
    logging.info("================================================"
                 "================================================")
    logging.info(
        "Running Zero-R classifier "
        "using %d-fold cross-validation.", CV_NUM)
    logging.info("================================================"
                 "================================================")
    total_accuracy = 0
    for i in range(CV_NUM):
        training_set, test_set = split_data(data, CV_NUM, i)
        classifier = zero_r_classifier(training_set, test_set)
        predictions = classifier.predict()
        logging.info(
            "Predictions in the form (number, predicted class): %s", predictions)
        accuracy = calculate_accuracy(test_set, predictions)
        total_accuracy += accuracy
    logging.info("Total accuracy: %s", total_accuracy / CV_NUM)


if __name__ == "__main__":
    main()
