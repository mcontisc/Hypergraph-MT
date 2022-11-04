import logging

logging.VERBOSE = 5
logging.addLevelName(logging.VERBOSE, "VERBOSE")
logging.Logger.verbose = lambda inst, msg, *args, **kwargs: inst.log(
    logging.VERBOSE, msg, *args, **kwargs
)
logging.LoggerAdapter.verbose = lambda inst, msg, *args, **kwargs: inst.log(
    logging.VERBOSE, msg, *args, **kwargs
)
logging.verbose = lambda msg, *args, **kwargs: logging.log(
    logging.VERBOSE, msg, *args, **kwargs
)


def setup_logging(logger_name=__name__, verbose=True, outfile=None):
    """
    SETUP LOGGING

    https://docs.python.org/3/howto/logging-cookbook.html
    """

    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - [PID %(process)4d] - %(name)-25s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)

    logger = logging.getLogger(logger_name)

    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if not logger.handlers:
        logger.addHandler(ch)

    # in case you want to write to file
    if outfile is not None:
        if outfile.endswith(".log") == False:
            outfile = outfile + ".log"
        fh = logging.FileHandler(outfile)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger
