import logging, sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler('m9_deliverables.log')])
logger = logging.getLogger(__name__)


def main():
    logger.info('M9 – Présentation finale et livrables (packaging)')
    # TODO: Rassembler outputs, générer rapport et slides (placeholders)
    pass


if __name__ == '__main__':
    main()
