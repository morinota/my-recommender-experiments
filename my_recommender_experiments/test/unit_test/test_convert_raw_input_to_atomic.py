from recommender_experiments.model.data_flow.convert_raw_input_to_atomic import ConvertRawInputToAtomicTask
import pandas as pd


def test_df_is_converted_str_for_atomic_file() -> None:
    # Arrange
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "category": ["cat1", "cat2"],
            "subcategory": ["subcat1", "subcat2"],
        }
    )
    field_type_by_id = {
        0: "id:token",
        1: "category:token",
        2: "subcategory:token",
    }
    sut = ConvertRawInputToAtomicTask()

    # Act
    atomic_data = sut.run(df, field_type_by_id)
    print("\n")
    print(atomic_data)

    # Assert
    atomic_data_expected = "id:token\tcategory:token\tsubcategory:token\n1\tcat1\tsubcat1\n2\tcat2\tsubcat2"
    assert atomic_data == atomic_data_expected
