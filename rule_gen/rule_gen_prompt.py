"""
Rule Generation Prompt Templates

This module provides prompt templates for generating Great Expectations rules
using LLMs.
"""

from typing import Union, Callable
from textwrap import dedent
import pandas as pd
from pathlib import Path


def get_expectation_list(list_id: int) -> str:
    """
    Get a list of available Great Expectations expectations.
    
    Args:
        list_id: Identifier for which list to return (0, 1, or 2)
        
    Returns:
        Formatted string of expectation names
    """
    expectations = {
        0: [
            "ExpectColumnDistinctValuesToBeInSet",
            "ExpectColumnDistinctValuesToContainSet",
            "ExpectColumnDistinctValuesToEqualSet",
            "ExpectColumnKLDivergenceToBeLessThan",
            "ExpectColumnMaxToBeBetween",
            "ExpectColumnMeanToBeBetween",
            "ExpectColumnMedianToBeBetween",
            "ExpectColumnMinToBeBetween",
            "ExpectColumnMostCommonValueToBeInSet",
            "ExpectColumnPairValuesAToBeGreaterThanB",
            "ExpectColumnPairValuesToBeEqual",
            "ExpectColumnPairValuesToBeInSet",
            "ExpectColumnProportionOfNonNullValuesToBeBetween",
            "ExpectColumnProportionOfUniqueValuesToBeBetween",
            "ExpectColumnQuantileValuesToBeBetween",
            "ExpectColumnStdevToBeBetween",
            "ExpectColumnSumToBeBetween",
            "ExpectColumnToExist",
            "ExpectColumnUniqueValueCountToBeBetween",
            "ExpectColumnValueLengthsToBeBetween",
            "ExpectColumnValueLengthsToEqual",
            "ExpectColumnValuesToBeBetween",
            "ExpectColumnValuesToBeDateutilParseable",
            "ExpectColumnValuesToBeDecreasing",
            "ExpectColumnValuesToBeIncreasing",
            "ExpectColumnValuesToBeInSet",
            "ExpectColumnValuesToBeInTypeList",
            "ExpectColumnValuesToBeJsonParseable",
            "ExpectColumnValuesToBeNull",
            "ExpectColumnValuesToBeOfType",
            "ExpectColumnValuesToBeUnique",
            "ExpectColumnValuesToMatchJsonSchema",
            "ExpectColumnValuesToMatchLikePattern",
            "ExpectColumnValuesToMatchLikePatternList",
            "ExpectColumnValuesToMatchRegex",
            "ExpectColumnValuesToMatchRegexList",
            "ExpectColumnValuesToMatchStrftimeFormat",
            "ExpectColumnValuesToNotBeInSet",
            "ExpectColumnValuesToNotBeNull",
            "ExpectColumnValuesToNotMatchLikePattern",
            "ExpectColumnValuesToNotMatchLikePatternList",
            "ExpectColumnValuesToNotMatchRegex",
            "ExpectColumnValuesToNotMatchRegexList",
            "ExpectColumnValueZScoresToBeLessThan",
            "ExpectCompoundColumnsToBeUnique",
            "ExpectMulticolumnSumToEqual",
            "ExpectMulticolumnValuesToBeUnique",
            "ExpectQueryResultsToMatchComparison",
            "ExpectSelectColumnValuesToBeUniqueWithinRecord",
            "ExpectTableColumnCountToBeBetween",
            "ExpectTableColumnCountToEqual",
            "ExpectTableColumnsToMatchOrderedList",
            "ExpectTableColumnsToMatchSet",
            "ExpectTableRowCountToBeBetween",
            "ExpectTableRowCountToEqual",
            "ExpectTableRowCountToEqualOtherTable",
            "UnexpectedRowsExpectation",
        ],
        1: [
            "ExpectColumnDistinctValuesToBeInSet(column: str, value_set: set-like)",
            "ExpectColumnDistinctValuesToContainSet(column: str, value_set: set-like)",
            "ExpectColumnDistinctValuesToEqualSet(column: str, value_set: set-like)",
            "ExpectColumnKLDivergenceToBeLessThan(column: str, partition_object: dict or None, threshold: float or None)",
            "ExpectColumnMaxToBeBetween(column: str, min_value: comparable type or None, max_value: comparable type or None, strict_min: boolean, strict_max: boolean)",
            "ExpectColumnMeanToBeBetween(column: str, min_value: float or None, max_value: float or None, strict_min: boolean, strict_max: boolean)",
            "ExpectColumnMedianToBeBetween(column: str, min_value: int or None, max_value: int or None, strict_min: boolean, strict_max: boolean)",
            "ExpectColumnMinToBeBetween(column: str, min_value: comparable type or None, max_value: comparable type or None, strict_min: boolean, strict_max: boolean)",
            "ExpectColumnMostCommonValueToBeInSet(column: str, value_set: set-like)",
            "ExpectColumnPairValuesAToBeGreaterThanB(column_A: str, column_B: str, or_equal: boolean or None)",
            "ExpectColumnPairValuesToBeEqual(column_A: str, column_B: str)",
            "ExpectColumnPairValuesToBeInSet(column_A: str, column_B: str, value_pairs_set: list of tuples)",
            "ExpectColumnProportionOfNonNullValuesToBeBetween(column: str, min_value: float or None, max_value: float or None, strict_min: boolean, strict_max: boolean)",
            "ExpectColumnProportionOfUniqueValuesToBeBetween(column: str, min_value: float or None, max_value: float or None, strict_min: boolean, strict_max: boolean)",
            "ExpectColumnQuantileValuesToBeBetween(column: str, quantile_ranges: dictionary with keys 'quantiles' and 'value_ranges', allow_relative_error: boolean or string)",
            "ExpectColumnStdevToBeBetween(column: str, min_value: float or None, max_value: float or None, strict_min: boolean, strict_max: boolean)",
            "ExpectColumnSumToBeBetween(column: str, min_value: comparable type or None, max_value: comparable type or None, strict_min: boolean, strict_max: boolean)",
            "ExpectColumnToExist(column: str, column_index: int or None, optional)",
            "ExpectColumnUniqueValueCountToBeBetween(column: str, min_value: int or None, max_value: int or None, strict_min: bool, strict_max: bool)",
            "ExpectColumnValueLengthsToBeBetween(column: str, min_value: int or None, max_value: int or None)",
            "ExpectColumnValueLengthsToEqual(column: str, value: int)",
            "ExpectColumnValueZScoresToBeLessThan(column: str, threshold: number)",
            "ExpectColumnValuesToBeBetween(column: str, min_value: comparable type or None, max_value: comparable type or None, strict_min: boolean, strict_max: boolean)",
            "ExpectColumnValuesToBeDateutilParseable(column: str)",
            "ExpectColumnValuesToBeDecreasing(column: str)",
            "ExpectColumnValuesToBeInSet(column: str, value_set: set-like)",
            "ExpectColumnValuesToBeInTypeList(column: str, type_list: list[str] or None)",
            "ExpectColumnValuesToBeIncreasing(column: str)",
            "ExpectColumnValuesToBeJsonParseable(column: str)",
            "ExpectColumnValuesToBeNull(column: str)",
            "ExpectColumnValuesToBeOfType(column: str, type_: None)",
            "ExpectColumnValuesToBeUnique(column: str)",
            "ExpectColumnValuesToMatchJsonSchema(column: str, json_schema: dict)",
            "ExpectColumnValuesToMatchLikePattern(column: str, like_pattern: str)",
            "ExpectColumnValuesToMatchLikePatternList(column: str, like_pattern_list: List[str])",
            "ExpectColumnValuesToMatchRegex(column: str, regex: str)",
            "ExpectColumnValuesToMatchRegexList(column: str, regex_list: list)",
            "ExpectColumnValuesToMatchStrftimeFormat(column: str, strftime_format: str or SuiteParameterDict)",
            "ExpectColumnValuesToNotBeInSet(column: str, value_set: set-like)",
            "ExpectColumnValuesToNotBeNull(column: str)",
            "ExpectColumnValuesToNotMatchLikePattern(column: str, like_pattern: str)",
            "ExpectColumnValuesToNotMatchLikePatternList(column: str, like_pattern_list: List[str])",
            "ExpectColumnValuesToNotMatchRegex(column: str, regex: str)",
            "ExpectColumnValuesToNotMatchRegexList(column: str, regex_list: list)",
            "ExpectCompoundColumnsToBeUnique(column_list: tuple or list)",
            "ExpectMulticolumnSumToEqual(column_list: tuple or list, sum_total: int or float)",
            "ExpectMulticolumnValuesToBeUnique(column_list: tuple or list)",
            "ExpectQueryResultsToMatchComparison()",
            "ExpectSelectColumnValuesToBeUniqueWithinRecord(column_list: tuple or list)",
            "ExpectTableColumnCountToBeBetween(min_value: int or None, max_value: int or None)",
            "ExpectTableColumnCountToEqual(value: int)",
            "ExpectTableColumnsToMatchOrderedList(column_list: list of str)",
            "ExpectTableColumnsToMatchSet(column_set: list of str, exact_match: boolean)",
            "ExpectTableRowCountToBeBetween(min_value: int or None, max_value: int or None, strict_min: boolean, strict_max: boolean)",
            "ExpectTableRowCountToEqual(value: int)",
            "ExpectTableRowCountToEqualOtherTable(other_table_name: str)",
            "UnexpectedRowsExpectation()",
        ],
        2: [
            "ExpectColumnDistinctValuesToBeInSet(column: str, value_set: set-like)",
            "ExpectColumnDistinctValuesToContainSet(column: str, value_set: set-like)",
            "ExpectColumnDistinctValuesToEqualSet(column: str, value_set: set-like)",
            "ExpectColumnKLDivergenceToBeLessThan(column: str, partition_object: dict or None, threshold: float or None)",
            "ExpectColumnMaxToBeBetween(column: str, min_value: comparable type or None, max_value: comparable type or None, strict_min: boolean, strict_max: boolean)",
            "ExpectColumnMeanToBeBetween(column: str, min_value: float or None, max_value: float or None, strict_min: boolean, strict_max: boolean)",
            "ExpectColumnMedianToBeBetween(column: str, min_value: int or None, max_value: int or None, strict_min: boolean, strict_max: boolean)",
            "ExpectColumnMinToBeBetween(column: str, min_value: comparable type or None, max_value: comparable type or None, strict_min: boolean, strict_max: boolean)",
            "ExpectColumnMostCommonValueToBeInSet(column: str, value_set: set-like)",
            "ExpectColumnPairValuesAToBeGreaterThanB(column_A: str, column_B: str, or_equal: boolean or None)",
            "ExpectColumnPairValuesToBeEqual(column_A: str, column_B: str)",
            "ExpectColumnPairValuesToBeInSet(column_A: str, column_B: str, value_pairs_set: list of tuples)",
            "ExpectColumnProportionOfNonNullValuesToBeBetween(column: str, min_value: float or None, max_value: float or None, strict_min: boolean, strict_max: boolean)",
            "ExpectColumnProportionOfUniqueValuesToBeBetween(column: str, min_value: float or None, max_value: float or None, strict_min: boolean, strict_max: boolean)",
            "ExpectColumnQuantileValuesToBeBetween(column: str, quantile_ranges: dictionary with keys 'quantiles' and 'value_ranges', allow_relative_error: boolean or string)",
            "ExpectColumnStdevToBeBetween(column: str, min_value: float or None, max_value: float or None, strict_min: boolean, strict_max: boolean)",
            "ExpectColumnSumToBeBetween(column: str, min_value: comparable type or None, max_value: comparable type or None, strict_min: boolean, strict_max: boolean)",
            "ExpectColumnToExist(column: str, column_index: int or None, optional)",
            "ExpectColumnUniqueValueCountToBeBetween(column: str, min_value: int or None, max_value: int or None, strict_min: bool, strict_max: bool)",
            "ExpectColumnValueLengthsToBeBetween(column: str, min_value: int or None, max_value: int or None)",
            "ExpectColumnValueLengthsToEqual(column: str, value: int)",
            "ExpectColumnValueZScoresToBeLessThan(column: str, threshold: number)",
            "ExpectColumnValuesToBeBetween(column: str, min_value: comparable type or None, max_value: comparable type or None, strict_min: boolean, strict_max: boolean)",
            "ExpectColumnValuesToBeDateutilParseable(column: str)",
            "ExpectColumnValuesToBeDecreasing(column: str)",
            "ExpectColumnValuesToBeInSet(column: str, value_set: set-like)",
            "ExpectColumnValuesToBeInTypeList(column: str, type_list: list[str] or None)",
            "ExpectColumnValuesToBeIncreasing(column: str)",
            "ExpectColumnValuesToBeJsonParseable(column: str)",
            "ExpectColumnValuesToBeNull(column: str)",
            "ExpectColumnValuesToBeOfType(column: str, type_: None)",
            "ExpectColumnValuesToBeUnique(column: str)",
            "ExpectColumnValuesToMatchJsonSchema(column: str, json_schema: dict)",
            "ExpectColumnValuesToMatchLikePattern(column: str, like_pattern: str)",
            "ExpectColumnValuesToMatchLikePatternList(column: str, like_pattern_list: List[str])",
            "ExpectColumnValuesToMatchRegex(column: str, regex: str)",
            "ExpectColumnValuesToMatchRegexList(column: str, regex_list: list)",
            "ExpectColumnValuesToMatchStrftimeFormat(column: str, strftime_format: str or SuiteParameterDict)",
            "ExpectColumnValuesToNotBeInSet(column: str, value_set: set-like)",
            "ExpectColumnValuesToNotBeNull(column: str)",
            "ExpectColumnValuesToNotMatchLikePattern(column: str, like_pattern: str)",
            "ExpectColumnValuesToNotMatchLikePatternList(column: str, like_pattern_list: List[str])",
            "ExpectColumnValuesToNotMatchRegex(column: str, regex: str)",
            "ExpectColumnValuesToNotMatchRegexList(column: str, regex_list: list)",
            "ExpectCompoundColumnsToBeUnique(column_list: tuple or list)",
            "ExpectMulticolumnSumToEqual(column_list: tuple or list, sum_total: int or float)",
            "ExpectMulticolumnValuesToBeUnique(column_list: tuple or list)",
            "ExpectQueryResultsToMatchComparison()",
            "ExpectSelectColumnValuesToBeUniqueWithinRecord(column_list: tuple or list)",
            "ExpectTableColumnCountToBeBetween(min_value: int or None, max_value: int or None)",
            "ExpectTableColumnCountToEqual(value: int)",
            "ExpectTableColumnsToMatchOrderedList(column_list: list of str)",
            "ExpectTableColumnsToMatchSet(column_set: list of str, exact_match: boolean)",
            "ExpectTableRowCountToBeBetween(min_value: int or None, max_value: int or None, strict_min: boolean, strict_max: boolean)",
            "ExpectTableRowCountToEqual(value: int)",
            "ExpectTableRowCountToEqualOtherTable(other_table_name: str)",
            "UnexpectedRowsExpectation()",
        ],
    }
    
    if list_id not in expectations:
        raise ValueError(f"Unknown expectation list ID: {list_id}. Use 0, 1, or 2.")
    
    return "\n".join(f"    {exp}" for exp in expectations[list_id])


def generate_prompt(
    clean_df: Union[pd.DataFrame, Path, str],
    polluted_df: Union[pd.DataFrame, Path, str],
    n_rows: int = 1,
    n_cols: int = 14,
    id_column: str = "pk",
    filter_criteria: Union[str, Callable[[pd.DataFrame], pd.Series], None] = None,
    prompt_id: int = 0,
    expectation_list_id: int = 0,
    error_detection_info: str = None,
) -> str:
    """
    Generate a prompt for LLM-based rule generation.
    
    Args:
        clean_df: Clean data as DataFrame or path to CSV file
        polluted_df: Polluted data as DataFrame or path to CSV file
        n_rows: Number of rows to include (default: 1 for single row)
        n_cols: Number of columns to include
        id_column: Column name used to align datasets
        filter_criteria: Optional filter string or callable
        prompt_id: Prompt template ID (0, 1, or 2)
        expectation_list_id: ID for which expectation list to include
        error_detection_info: Additional info for error detection
        
    Returns:
        Formatted prompt string
    """
    # Read data if paths are provided
    if isinstance(clean_df, (str, Path)):
        clean_df = pd.read_csv(clean_df).sort_values(id_column)
    if isinstance(polluted_df, (str, Path)):
        polluted_df = pd.read_csv(polluted_df).sort_values(id_column)
    
    # Apply filter if provided
    if filter_criteria is not None:
        if isinstance(filter_criteria, str):
            polluted_df = polluted_df.query(filter_criteria)
        elif callable(filter_criteria):
            mask = filter_criteria(polluted_df)
            polluted_df = polluted_df[mask]
    
    # Align datasets on ID
    matched_ids = polluted_df[id_column].isin(clean_df[id_column])
    polluted_df = polluted_df[matched_ids]
    clean_df = clean_df[clean_df[id_column].isin(polluted_df[id_column])]
    
    # Get error information from first row
    err_col_val = polluted_df["errCol"].iloc[0].split(";")[1:]
    err_rule_val = polluted_df["errRule"].iloc[0].split(";")[1:]
    
    error_info_lines = []
    for i, (err_col, err_rule) in enumerate(zip(err_col_val, err_rule_val)):
        if "<->" in err_col:
            error_info_lines.append(f"Columns {' and '.join(err_col.split('<->'))} have error: {err_rule}.")
        else:
            error_info_lines.append(f"Column {err_col} has error: {err_rule}.")
    
    error_info = "\n".join(error_info_lines)
    
    # Truncate to requested columns
    clean_df = clean_df.iloc[:n_rows, :n_cols]
    polluted_df = polluted_df.iloc[:n_rows, :n_cols]
    
    # Convert to markdown tables
    clean_md = clean_df.to_markdown(index=False)
    polluted_md = polluted_df.to_markdown(index=False)
    
    # Default error detection info for Austrian postal data
    if error_detection_info is None:
        error_detection_info = """Du hast die Rolle eines Postbeamter und must die korrektheit der Daten einer Person überprüfen. Betrachte alle Felder zusammen. Der Telefonnummer kann es sich auch um eine Mobiltelefonnummer handeln. Sei vorsichtig manche Orte wirken wie abkürzungen sind aber korrekte Orte in Österreich. Wenn ein Ort/Stadt nicht eindeutig ist nutze das Feld Postleitzahl um die Eindeutigkeit zu verifizieren. Wenn die Postleitzahl nicht im Ort/Stadt passt dann ist eines der beiden Felder inkorrekt. Wenn die Kombination von Postleitzahl und Ort/Stadt korrekt und eindeutig ist benötigt es keine weitere information. Jedes Feld darf nur die relevante Information enthalten nicht mehr."""
    
    # Generate prompt based on ID
    if prompt_id == 0:
        prompt = dedent(f"""
            Hi I have the following Data with an error. Generate Great Expectation expectations that flag the data with the errors as invalid. 
            I am using Version 1.9.1.
            The expectations generated by you will be part of a suite of already existing expectations that checks a larger data.frame.
            You do not need to concern yourself with expectations that already exist.
            Data with Error:
            {polluted_md}
            The same Data cleaned:
            {clean_md}
            # Information for the Error Detection:
            {error_detection_info}
            # Information about the Syntax of 'Great Expectation' Expectations
            ## batch
            This is how I want to use the code you create:
            suite = gx.ExpectationSuite(name="expectation_suite_name")
            <Additional Code run here, for example if a custom class is needed.>
            suite.add_expectation(<Code generated by you>)
            batch.validate(suite)
            You should also consider that all values are of the type string. Even those look numeric, e.g. plz!
            Preferabel use gx.expectations that exists for great expectation 1.9.1. Name and description are not valid parameter. Here is a list of those:
            {get_expectation_list(expectation_list_id)}
            ## Custom Expectation
            If you need to create a own Expectation here is a information to help you:
            Currently registering a ColumnMapExpectation with an own _validation function is buggy and dose not work reliably. So do not use it! 
            ### Sample Expectation:
            Be careful do not add a **init**() function where you set the values as those are not allowed. If you want to create an Expectation please make sure to provide the code correctly as additional code. Be careful only: great expectation, pandas and standard python packages are available. No Way to install more.
            class ExpectValidPassengerCount(gx.expectations.ExpectColumnValuesToBeBetween):
            column: str = "pk"
            min_value: int = 100
            max_value: int = 400
            description: str = "There should be between 1 and 6 passengers."
            # Task
            1) Detect the Errors and list them.
            2) Find general rules but only for the error you found in the provided polluted data.
            3) create great expectation Expectations. Check the Syntax carefully.
            4) return your answer with all relevant Rules into this json format and write '# Final Json Format' before this. Name, column and code must be filled out. The additional code only if necessary. Do not any additonal fields to the json as they will be ignored!:
            ```json
            {{
                "great_expectations_final":{{[
                    "<expectation_n>":{{
                    "name":"<expectation_name>",
                    "column":"<Column>",
                    "code":"<Code generated by you and executed inside suite.add_expectation(). Make sure this code can be run as is. >",
                    "additional_code":"<Additional Code run here, for example if a custom class is needed. Make sure to include all the necessary code! This also means any variable you define that are needed. >"
                    "datatype":"string",
                    "rule_description": "<Generate a description what this rule should do.>"
                    }}
                ]}}
            }}
            ```
            """)
    elif prompt_id == 1:
        prompt = dedent(f"""
            Hi I have the following Data with an error. Generate Great Expectation expectations that flag the data with the errors as invalid. 
            I am using Version 1.9.1.
            The expectations generated by you will be part of a suite of already existing expectations that checks a larger data.frame.
            You do not need to concern yourself with expectations that already exist.
            Data with Error:
            {polluted_md}
            # Information for the Error Detection:
            {error_detection_info}
            # Information about the Syntax of 'Great Expectation' Expectations
            ## batch
            This is how I want to use the code you create:
            suite = gx.ExpectationSuite(name="expectation_suite_name")
            <Additional Code run here, for example if a custom class is needed.>
            suite.add_expectation(<Code generated by you>)
            batch.validate(suite)
            You should also consider that all values are of the type string. Even those look numeric, e.g. plz!
            Preferabel use gx.expectations that exists for great expectation 1.9.1. Name and description are not valid parameter. Here is a list of those:
            {get_expectation_list(expectation_list_id)}
            ## Custom Expectation
            If you need to create a own Expectation here is a information to help you:
            Currently registering a ColumnMapExpectation with an own _validation function is buggy and dose not work reliably. So do not use it! 
            ### Sample Expectation:
            Be careful do not add a **init**() function where you set the values as those are not allowed. If you want to create an Expectation please make sure to provide the code correctly as additional code. Be careful only: great expectation, pandas and standard python packages are available. No Way to install more.
            class ExpectValidPassengerCount(gx.expectations.ExpectColumnValuesToBeBetween):
            column: str = "pk"
            min_value: int = 100
            max_value: int = 400
            description: str = "There should be between 1 and 6 passengers."
            # Task
            1) Detect the Errors and list them.
            2) Find general rules but only for the error you found in the provided polluted data.
            3) create great expectation Expectations. Check the Syntax carefully.
            4) return your answer with all relevant Rules into this json format and write '# Final Json Format' before this. Name, column and code must be filled out. The additional code only if necessary. Do not any additonal fields to the json as they will be ignored!:
            ```json
            {{
                "great_expectations_final":{{[
                    "<expectation_n>":{{
                    "name":"<expectation_name>",
                    "column":"<Column>",
                    "code":"<Code generated by you and executed inside suite.add_expectation(). Make sure this code can be run as is. >",
                    "additional_code":"<Additional Code run here, for example if a custom class is needed. Make sure to include all the necessary code! This also means any variable you define that are needed. >"
                    "datatype":"string",
                    "rule_description": "<Generate a description what this rule should do.>"
                    }}
                ]}}
            }}
            ```
            """)
    elif prompt_id == 2:
        prompt = dedent(f"""
            Hi I have the following Data with an error. Generate Great Expectation expectations that flag the data with the errors as invalid. 
            I am using Version 1.9.1.
            The expectations generated by you will be part of a suite of already existing expectations that checks a larger data.frame.
            You do not need to concern yourself with expectations that already exist.
            Data with Error:
            {polluted_md}
            # Information for the Error Detection:
            {error_detection_info}
            Here are some information to help you find the errors: 
            {error_info}
            # Information about the Syntax of 'Great Expectation' Expectations
            ## batch
            This is how I want to use the code you create:
            suite = gx.ExpectationSuite(name="expectation_suite_name")
            <Additional Code run here, for example if a custom class is needed.>
            suite.add_expectation(<Code generated by you>)
            batch.validate(suite)
            You should also consider that all values are of the type string. Even those look numeric, e.g. plz!
            Preferabel use gx.expectations that exists for great expectation 1.9.1. Name and description are not valid parameter. Here is a list of those:
            {get_expectation_list(expectation_list_id)}
            ## Custom Expectation
            If you need to create a own Expectation here is a information to help you:
            Currently registering a ColumnMapExpectation with an own _validation function is buggy and dose not work reliably. So do not use it! 
            ### Sample Expectation:
            Be careful do not add a **init**() function where you set the values as those are not allowed. If you want to create an Expectation please make sure to provide the code correctly as additional code. Be careful only: great expectation, pandas and standard python packages are available. No Way to install more.
            class ExpectValidPassengerCount(gx.expectations.ExpectColumnValuesToBeBetween):
            column: str = "pk"
            min_value: int = 100
            max_value: int = 400
            description: str = "There should be between 1 and 6 passengers."
            # Task
            1) Detect the Errors and list them.
            2) Find general rules but only for the error you found in the provided polluted data.
            3) create great expectation Expectations. Check the Syntax carefully.
            4) return your answer with all relevant Rules into this json format and write '# Final Json Format' before this. Name, column and code must be filled out. The additional code only if necessary. Do not any additonal fields to the json as they will be ignored!:
            ```json
            {{
                "great_expectations_final":{{[
                    "<expectation_n>":{{
                    "name":"<expectation_name>",
                    "column":"<Column>",
                    "code":"<Code generated by you and executed inside suite.add_expectation(). Make sure this code can be run as is. >",
                    "additional_code":"<Additional Code run here, for example if a custom class is needed. Make sure to include all the necessary code! This also means any variable you define that are needed. >"
                    "datatype":"string",
                    "rule_description": "<Generate a description what this rule should do.>"
                    }}
                ]}}
            }}
            ```
            """)
    else:
        raise ValueError(f"Unsupported prompt_id: {prompt_id}. Use 0, 1, or 2.")
    
    return prompt.strip()


def main():
    
if __name__ == "__main__":
    main()