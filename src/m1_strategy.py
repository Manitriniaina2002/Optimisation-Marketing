import logging, sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('m1_strategy.log')])
logger = logging.getLogger(__name__)


def main():
    logger.info('M1 – Compréhension des enjeux stratégiques')
    # TODO: Rédiger SWOT, 5P, parcours client
    # Exporter une note synthèse dans reports/
    pass


if __name__ == '__main__':
    main()
