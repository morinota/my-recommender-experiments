import os
import sys

from convert_raw_input_to_atomic import ConvertRawInputToAtomicTask
import pandas as pd


def test_df_is_converted_str_for_atomic_file() -> None:
    # Arrange
    print(os.getcwd)
    # import可能なpathを出力
    print(sys.path)
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "category": ["cat1", "cat2"],
            "subcategory": ["subcat1", "subcat2"],
        }
    )
    feature_type_by_name = {
        "id": "token",
        "category": ":token",
        "subcategory": "token",
    }
    sut = ConvertRawInputToAtomicTask()

    # Act
    atomic_data = sut.run(df, feature_type_by_name)
    print("\n")
    print(atomic_data)

    # Assert
    atomic_data_expected = "id:token\tcategory:token\tsubcategory:token\n1\tcat1\tsubcat1\n2\tcat2\tsubcat2"
    assert atomic_data == atomic_data_expected
