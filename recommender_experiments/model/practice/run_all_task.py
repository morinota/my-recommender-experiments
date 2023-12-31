from calc_accuracy import Step54CalcAccuracyTask
from calc_confusion_matrix import Step55CalcConfusionMatrixTask
import gokart


class RunAllTask(gokart.TaskOnKart):
    """タスクが全部実行されるようにするためのタスク。"""

    task_list = [
        Step54CalcAccuracyTask(),
        Step55CalcConfusionMatrixTask(),
    ]

    def requires(self):
        return {str(i): task for i, task in enumerate(RunAllTask.task_list)}


if __name__ == "__main__":
    gokart.run(["RunAllTask", "--local-scheduler", "--rerun"])
    # luigid で動かす
    # gokart.run(['RunAllTask', '--rerun'])
    # タスクの依存関係を出力する
    # print(gokart.make_task_info_as_tree_str(RunAllTask()))
