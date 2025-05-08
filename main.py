import argparse
from data_processing import load_and_clean_data
from feature_engineering import build_features
from modeling import train_model
from evaluation import evaluate_model


def main():
    choices = ['preprocess', 'features', 'train', 'evaluate', 'all']
    parser = argparse.ArgumentParser()
    parser.add_argument('step', choices=choices, help='Pipeline stage to run')
    args = parser.parse_args()

    if args.step in ('preprocess', 'all'):
        load_and_clean_data()
    if args.step in ('features', 'all'):
        build_features()
    if args.step in ('train', 'all'):
        train_model()
    if args.step in ('evaluate', 'all'):
        evaluate_model()


if __name__ == '__main__':
    main()