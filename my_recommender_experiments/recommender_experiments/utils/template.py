from logging import getLogger

import gokart

logger = getLogger(__name__)


class GokartTask(gokart.TaskOnKart):
    task_namespace = 'recommender_experiments'
