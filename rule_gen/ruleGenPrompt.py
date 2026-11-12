from typing import Union, Callable
from textwrap import dedent
from itertools import zip_longest
import pandas as pd
from pathlib import Path


def _split_error_labels(value):
    """Split the semicolon-delimited error metadata, preserving empty positions."""
    if value is None or pd.isna(value):
        return []

    parts = [part.strip() for part in str(value).split(";")]
    while parts and not parts[0]:
        parts.pop(0)
    while parts and not parts[-1]:
        parts.pop()
    return parts


def _build_error_hints(polluted_df):
    """Describe every labeled error in the first row supplied to the prompt."""
    class_hints = []
    location_hints = []

    if polluted_df.empty:
        no_labels_message = "No injected-error labels are available for the displayed row."
        return no_labels_message, no_labels_message

    row = polluted_df.iloc[0]
    error_columns = _split_error_labels(row.get("errCol"))
    error_rules = _split_error_labels(row.get("errRule"))
    for column, error_rule in zip_longest(error_columns, error_rules, fillvalue=""):
        if not column:
            continue
        error_rule = error_rule or "unspecified"

        if "<->" in column:
            affected_columns = [name.strip() for name in column.split("<->") if name.strip()]
            formatted_columns = " and ".join(affected_columns)
            class_hints.append(f"Columns {formatted_columns} are {error_rule}.")
            location_hints.append(f"Columns {formatted_columns} have an error.")
        else:
            class_hints.append(f"Column {column} has the error {error_rule}.")
            location_hints.append(f"Column {column} have an error.")

    no_labels_message = "No injected-error labels are available for the displayed row."
    return (
        "\n".join(class_hints) if class_hints else no_labels_message,
        "\n".join(location_hints) if location_hints else no_labels_message,
    )

def getExpectationLi(liId):
    expectationLiStr = ""
    if liId == 0:
        expectationLiStr = dedent(f"""
            ExpectColumnDistinctValuesToBeInSet,
            ExpectColumnDistinctValuesToContainSet,
            ExpectColumnDistinctValuesToEqualSet,
            ExpectColumnKLDivergenceToBeLessThan,
            ExpectColumnMaxToBeBetween,
            ExpectColumnMeanToBeBetween,
            ExpectColumnMedianToBeBetween,
            ExpectColumnMinToBeBetween,
            ExpectColumnMostCommonValueToBeInSet,
            ExpectColumnPairValuesAToBeGreaterThanB,
            ExpectColumnPairValuesToBeEqual,
            ExpectColumnPairValuesToBeInSet,
            ExpectColumnProportionOfNonNullValuesToBeBetween,
            ExpectColumnProportionOfUniqueValuesToBeBetween,
            ExpectColumnQuantileValuesToBeBetween,
            ExpectColumnStdevToBeBetween,
            ExpectColumnSumToBeBetween,
            ExpectColumnToExist,
            ExpectColumnUniqueValueCountToBeBetween,
            ExpectColumnValueLengthsToBeBetween,
            ExpectColumnValueLengthsToEqual,
            ExpectColumnValuesToBeBetween,
            ExpectColumnValuesToBeDateutilParseable,
            ExpectColumnValuesToBeDecreasing,
            ExpectColumnValuesToBeIncreasing,
            ExpectColumnValuesToBeInSet,
            ExpectColumnValuesToBeInTypeList,
            ExpectColumnValuesToBeJsonParseable,
            ExpectColumnValuesToBeNull,
            ExpectColumnValuesToBeOfType,
            ExpectColumnValuesToBeUnique,
            ExpectColumnValuesToMatchJsonSchema,
            ExpectColumnValuesToMatchLikePattern,
            ExpectColumnValuesToMatchLikePatternList,
            ExpectColumnValuesToMatchRegex,
            ExpectColumnValuesToMatchRegexList,
            ExpectColumnValuesToMatchStrftimeFormat,
            ExpectColumnValuesToNotBeInSet,
            ExpectColumnValuesToNotBeNull,
            ExpectColumnValuesToNotMatchLikePattern,
            ExpectColumnValuesToNotMatchLikePatternList,
            ExpectColumnValuesToNotMatchRegex,
            ExpectColumnValuesToNotMatchRegexList,
            ExpectColumnValueZScoresToBeLessThan,
            ExpectCompoundColumnsToBeUnique,
            ExpectMulticolumnSumToEqual,
            ExpectMulticolumnValuesToBeUnique,
            ExpectQueryResultsToMatchComparison,
            ExpectSelectColumnValuesToBeUniqueWithinRecord,
            ExpectTableColumnCountToBeBetween,
            ExpectTableColumnCountToEqual,
            ExpectTableColumnsToMatchOrderedList,
            ExpectTableColumnsToMatchSet,
            ExpectTableRowCountToBeBetween,
            ExpectTableRowCountToEqual,
            ExpectTableRowCountToEqualOtherTable,
            UnexpectedRowsExpectation""")
    elif liId == 1:#
        expectationLiStr = dedent(f"""
            ExpectColumnDistinctValuesToBeInSet(column: str, value_set: set-like)
            ExpectColumnDistinctValuesToContainSet(column: str, value_set: set-like)
            ExpectColumnDistinctValuesToEqualSet(column: str, value_set: set-like)
            ExpectColumnKLDivergenceToBeLessThan(column: str, partition_object: dict or None, threshold: float or None)
            ExpectColumnMaxToBeBetween(column: str, min_value: comparable type or None, max_value: comparable type or None, strict_min: boolean, strict_max: boolean)
            ExpectColumnMeanToBeBetween(column: str, min_value: float or None, max_value: float or None, strict_min: boolean, strict_max: boolean)
            ExpectColumnMedianToBeBetween(column: str, min_value: int or None, max_value: int or None, strict_min: boolean, strict_max: boolean)
            ExpectColumnMinToBeBetween(column: str, min_value: comparable type or None, max_value: comparable type or None, strict_min: boolean, strict_max: boolean)
            ExpectColumnMostCommonValueToBeInSet(column: str, value_set: set-like)
            ExpectColumnPairValuesAToBeGreaterThanB(column_A: str, column_B: str, or_equal: boolean or None)
            ExpectColumnPairValuesToBeEqual(column_A: str, column_B: str)
            ExpectColumnPairValuesToBeInSet(column_A: str, column_B: str, value_pairs_set: list of tuples)
            ExpectColumnProportionOfNonNullValuesToBeBetween(column: str, min_value: float or None, max_value: float or None, strict_min: boolean, strict_max: boolean)
            ExpectColumnProportionOfUniqueValuesToBeBetween(column: str, min_value: float or None, max_value: float or None, strict_min: boolean, strict_max: boolean)
            ExpectColumnQuantileValuesToBeBetween(column: str, quantile_ranges: dictionary with keys 'quantiles' and 'value_ranges', allow_relative_error: boolean or string)
            ExpectColumnStdevToBeBetween(column: str, min_value: float or None, max_value: float or None, strict_min: boolean, strict_max: boolean)
            ExpectColumnSumToBeBetween(column: str, min_value: comparable type or None, max_value: comparable type or None, strict_min: boolean, strict_max: boolean)
            ExpectColumnToExist(column: str, column_index: int or None, optional)
            ExpectColumnUniqueValueCountToBeBetween(column: str, min_value: int or None, max_value: int or None, strict_min: bool, strict_max: bool)
            ExpectColumnValueLengthsToBeBetween(column: str, min_value: int or None, max_value: int or None)
            ExpectColumnValueLengthsToEqual(column: str, value: int)
            ExpectColumnValueZScoresToBeLessThan(column: str, threshold: number)
            ExpectColumnValuesToBeBetween(column: str, min_value: comparable type or None, max_value: comparable type or None, strict_min: boolean, strict_max: boolean)
            ExpectColumnValuesToBeDateutilParseable(column: str)
            ExpectColumnValuesToBeDecreasing(column: str)
            ExpectColumnValuesToBeInSet(column: str, value_set: set-like)
            ExpectColumnValuesToBeInTypeList(column: str, type_list: list[str] or None)
            ExpectColumnValuesToBeIncreasing(column: str)
            ExpectColumnValuesToBeJsonParseable(column: str)
            ExpectColumnValuesToBeNull(column: str)
            ExpectColumnValuesToBeOfType(column: str, type_: None)
            ExpectColumnValuesToBeUnique(column: str)
            ExpectColumnValuesToMatchJsonSchema(column: str, json_schema: dict)
            ExpectColumnValuesToMatchLikePattern(column: str, like_pattern: str)
            ExpectColumnValuesToMatchLikePatternList(column: str, like_pattern_list: List[str])
            ExpectColumnValuesToMatchRegex(column: str, regex: str)
            ExpectColumnValuesToMatchRegexList(column: str, regex_list: list)
            ExpectColumnValuesToMatchStrftimeFormat(column: str, strftime_format: str or SuiteParameterDict)
            ExpectColumnValuesToNotBeInSet(column: str, value_set: set-like)
            ExpectColumnValuesToNotBeNull(column: str)
            ExpectColumnValuesToNotMatchLikePattern(column: str, like_pattern: str)
            ExpectColumnValuesToNotMatchLikePatternList(column: str, like_pattern_list: List[str])
            ExpectColumnValuesToNotMatchRegex(column: str, regex: str)
            ExpectColumnValuesToNotMatchRegexList(column: str, regex_list: list)
            ExpectCompoundColumnsToBeUnique(column_list: tuple or list)
            ExpectMulticolumnSumToEqual(column_list: tuple or list, sum_total: int or float)
            ExpectMulticolumnValuesToBeUnique(column_list: tuple or list)
            ExpectQueryResultsToMatchComparison()
            ExpectSelectColumnValuesToBeUniqueWithinRecord(column_list: tuple or list)
            ExpectTableColumnCountToBeBetween(min_value: int or None, max_value: int or None)
            ExpectTableColumnCountToEqual(value: int)
            ExpectTableColumnsToMatchOrderedList(column_list: list of str)
            ExpectTableColumnsToMatchSet(column_set: list of str, exact_match: boolean)
            ExpectTableRowCountToBeBetween(min_value: int or None, max_value: int or None, strict_min: boolean, strict_max: boolean)
            ExpectTableRowCountToEqual(value: int)
            ExpectTableRowCountToEqualOtherTable(other_table_name: str)
            UnexpectedRowsExpectation()
            """)
    elif liId == 2:#
        expectationLiStr = dedent(f"""
            ExpectColumnDistinctValuesToBeInSet(column: str, value_set: set-like)
            ExpectColumnDistinctValuesToContainSet(column: str, value_set: set-like)
            ExpectColumnDistinctValuesToEqualSet(column: str, value_set: set-like)
            ExpectColumnKLDivergenceToBeLessThan(column: str, partition_object: dict or None, threshold: float or None)
            ExpectColumnMaxToBeBetween(column: str, min_value: comparable type or None, max_value: comparable type or None, strict_min: boolean, strict_max: boolean)
            ExpectColumnMeanToBeBetween(column: str, min_value: float or None, max_value: float or None, strict_min: boolean, strict_max: boolean)
            ExpectColumnMedianToBeBetween(column: str, min_value: int or None, max_value: int or None, strict_min: boolean, strict_max: boolean)
            ExpectColumnMinToBeBetween(column: str, min_value: comparable type or None, max_value: comparable type or None, strict_min: boolean, strict_max: boolean)
            ExpectColumnMostCommonValueToBeInSet(column: str, value_set: set-like)
            ExpectColumnPairValuesAToBeGreaterThanB(column_A: str, column_B: str, or_equal: boolean or None)
            ExpectColumnPairValuesToBeEqual(column_A: str, column_B: str)
            ExpectColumnPairValuesToBeInSet(column_A: str, column_B: str, value_pairs_set: list of tuples)
            ExpectColumnProportionOfNonNullValuesToBeBetween(column: str, min_value: float or None, max_value: float or None, strict_min: boolean, strict_max: boolean)
            ExpectColumnProportionOfUniqueValuesToBeBetween(column: str, min_value: float or None, max_value: float or None, strict_min: boolean, strict_max: boolean)
            ExpectColumnQuantileValuesToBeBetween(column: str, quantile_ranges: dictionary with keys 'quantiles' and 'value_ranges', allow_relative_error: boolean or string)
            ExpectColumnStdevToBeBetween(column: str, min_value: float or None, max_value: float or None, strict_min: boolean, strict_max: boolean)
            ExpectColumnSumToBeBetween(column: str, min_value: comparable type or None, max_value: comparable type or None, strict_min: boolean, strict_max: boolean)
            ExpectColumnToExist(column: str, column_index: int or None, optional)
            ExpectColumnUniqueValueCountToBeBetween(column: str, min_value: int or None, max_value: int or None, strict_min: bool, strict_max: bool)
            ExpectColumnValueLengthsToBeBetween(column: str, min_value: int or None, max_value: int or None)
            ExpectColumnValueLengthsToEqual(column: str, value: int)
            ExpectColumnValueZScoresToBeLessThan(column: str, threshold: number)
            ExpectColumnValuesToBeBetween(column: str, min_value: comparable type or None, max_value: comparable type or None, strict_min: boolean, strict_max: boolean)
            ExpectColumnValuesToBeDateutilParseable(column: str)
            ExpectColumnValuesToBeDecreasing(column: str)
            ExpectColumnValuesToBeInSet(column: str, value_set: set-like)
            ExpectColumnValuesToBeInTypeList(column: str, type_list: list[str] or None)
            ExpectColumnValuesToBeIncreasing(column: str)
            ExpectColumnValuesToBeJsonParseable(column: str)
            ExpectColumnValuesToBeNull(column: str)
            ExpectColumnValuesToBeOfType(column: str, type_: None)
            ExpectColumnValuesToBeUnique(column: str)
            ExpectColumnValuesToMatchJsonSchema(column: str, json_schema: dict)
            ExpectColumnValuesToMatchLikePattern(column: str, like_pattern: str)
            ExpectColumnValuesToMatchLikePatternList(column: str, like_pattern_list: List[str])
            ExpectColumnValuesToMatchRegex(column: str, regex: str)
            ExpectColumnValuesToMatchRegexList(column: str, regex_list: list)
            ExpectColumnValuesToMatchStrftimeFormat(column: str, strftime_format: str or SuiteParameterDict)
            ExpectColumnValuesToNotBeInSet(column: str, value_set: set-like)
            ExpectColumnValuesToNotBeNull(column: str)
            ExpectColumnValuesToNotMatchLikePattern(column: str, like_pattern: str)
            ExpectColumnValuesToNotMatchLikePatternList(column: str, like_pattern_list: List[str])
            ExpectColumnValuesToNotMatchRegex(column: str, regex: str)
            ExpectColumnValuesToNotMatchRegexList(column: str, regex_list: list)
            ExpectCompoundColumnsToBeUnique(column_list: tuple or list)
            ExpectMulticolumnSumToEqual(column_list: tuple or list, sum_total: int or float)
            ExpectMulticolumnValuesToBeUnique(column_list: tuple or list)
            ExpectQueryResultsToMatchComparison()
            ExpectSelectColumnValuesToBeUniqueWithinRecord(column_list: tuple or list)
            ExpectTableColumnCountToBeBetween(min_value: int or None, max_value: int or None)
            ExpectTableColumnCountToEqual(value: int)
            ExpectTableColumnsToMatchOrderedList(column_list: list of str)
            ExpectTableColumnsToMatchSet(column_set: list of str, exact_match: boolean)
            ExpectTableRowCountToBeBetween(min_value: int or None, max_value: int or None, strict_min: boolean, strict_max: boolean)
            ExpectTableRowCountToEqual(value: int)
            ExpectTableRowCountToEqualOtherTable(other_table_name: str)
            UnexpectedRowsExpectation()
            """)
    return expectationLiStr

def generate_prompt(clean_csv, polluted_csv, n_rows=5, n_cols=14, id_column: str = "pk", 
                    filter_criteria: Union[str, Callable[[pd.DataFrame], pd.Series], None] = None, selected_pk= None,
                    promptId=11,fileContent=None,expectationLiId=0, err_dect_info = None):
    """
    Generates a Markdown prompt showing clean vs polluted data samples side by side,
    with optional filtering and row alignment via an ID column.

    Args:
        clean_csv (str): Path to the clean data CSV file.
        polluted_csv (str): Path to the polluted data CSV file.
        id_column (str): Column name used to align datasets.
        n_rows (int): Number of rows to include in the output.
        n_cols (int): Number of columns to include in each table.
        filter_criteria (str or callable, optional):
            - If str, it is passed to `DataFrame.query()`.
            - If callable, it should accept a DataFrame and return a boolean mask.
            - If None, no filtering is applied.
        err_dect_info (str): data Information needed for error detection, default None --> leads to info for PCI beeing used
        promptId (int): Paper prompt variant: 8 (Dirty & Clean),
            9 (Just Dirty), 10 (Dirty & Type), or 11 (Dirty & Column).

    Returns:
        str: A formatted Markdown prompt string.
    """

    if isinstance(clean_csv,Path):
        clean_df = pd.read_csv(clean_csv).sort_values('pk')
    elif isinstance(clean_csv,pd.DataFrame) or isinstance(clean_csv,pd.Series):
        clean_df = clean_csv

    if isinstance(polluted_csv,Path):
        polluted_df = pd.read_csv(polluted_csv).sort_values('pk')
    elif isinstance(polluted_csv,pd.DataFrame) or isinstance(polluted_csv,pd.Series):
        polluted_df = polluted_csv

    
    if filter_criteria is not None:
        if isinstance(filter_criteria, str):
            polluted_df = polluted_df.query(filter_criteria)
        elif callable(filter_criteria):
            mask = filter_criteria(polluted_df)
            polluted_df = polluted_df[mask]
        else:
            raise ValueError("filter_criteria must be a string or callable.")

    matched_ids = polluted_df[id_column].isin(clean_df[id_column])
    polluted_df = polluted_df[matched_ids]
    clean_df = clean_df[clean_df[id_column].isin(polluted_df[id_column])]

    errClassInfo, errLocInfo = _build_error_hints(polluted_df)


    clean_df = clean_df.iloc[:n_rows, :n_cols]
    idVals = polluted_df[id_column].iloc[:n_rows].values
    polluted_df = polluted_df.iloc[:n_rows, :n_cols]

    clean_md = clean_df.to_markdown(index=False)
    polluted_md = polluted_df.to_markdown(index=False)

    if err_dect_info is None:
        err_dect_info = """Du hast die Rolle eines Postbeamter und must die korrektheit der Daten einer Person überprüfen. Betrachte alle Felder zusammen. Der Telefonnummer kann es sich auch um eine Mobiltelefonnummer handeln. Sei vorsichtig manche Orte wirken wie abkürzungen sind aber korrekte Orte in Österreich. Wenn ein Ort/Stadt nicht eindeutig ist nutze das Feld Postleitzahl um die Eindeutigkeit zu verifizieren. Wenn die Postleitzahl nicht im Ort/Stadt passt dann ist eines der beiden Felder inkorrekt. Wenn die Kombination von Postleitzahl und Ort/Stadt korrekt und eindeutig ist benötigt es keine weitere information. Jedes Feld darf nur die relevante Information enthalten nicht mehr.
            I provided the same data with and without the error to help you find the error. """


    

    if promptId == 8:
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
            {err_dect_info}
            # Information about the Syntax of 'Great Expectation' Expectations
            ## batch
            This is how I want to use the code you create:
            suite = gx.ExpectationSuite(name="expectation_suite_name")
            <Additional Code run here, for example if a custom class is needed.>
            suite.add_expectation(<Code generated by you>)
            batch.validate(suite)
            You should also consider that all values are of the type string. Even those look numeric, e.g. plz!
            Preferabel use gx.expectations that exists for great expectation 1.9.1. Name and description are not valid parameter. Here is a list of those:
            {getExpectationLi(expectationLiId)}
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
    elif promptId == 9:
        prompt = dedent(f"""
            Hi I have the following Data with an error. Generate Great Expectation expectations that flag the data with the errors as invalid. 
            I am using Version 1.9.1.
            The expectations generated by you will be part of a suite of already existing expectations that checks a larger data.frame.
            You do not need to concern yourself with expectations that already exist.
            Data with Error:
            {polluted_md}
            # Information for the Error Detection:
            {err_dect_info}
            # Information about the Syntax of 'Great Expectation' Expectations
            ## batch
            This is how I want to use the code you create:
            suite = gx.ExpectationSuite(name="expectation_suite_name")
            <Additional Code run here, for example if a custom class is needed.>
            suite.add_expectation(<Code generated by you>)
            batch.validate(suite)
            You should also consider that all values are of the type string. Even those look numeric, e.g. plz!
            Preferabel use gx.expectations that exists for great expectation 1.9.1. Name and description are not valid parameter. Here is a list of those:
            {getExpectationLi(expectationLiId)}
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
    elif promptId == 10:
        prompt = dedent(f"""
            Hi I have the following Data with an error. Generate Great Expectation expectations that flag the data with the errors as invalid. 
            I am using Version 1.9.1.
            The expectations generated by you will be part of a suite of already existing expectations that checks a larger data.frame.
            You do not need to concern yourself with expectations that already exist.
            Data with Error:
            {polluted_md}
            # Information for the Error Detection:
            {err_dect_info}
            Here are some information to help you find the errors: 
            {errClassInfo}
            # Information about the Syntax of 'Great Expectation' Expectations
            ## batch
            This is how I want to use the code you create:
            suite = gx.ExpectationSuite(name="expectation_suite_name")
            <Additional Code run here, for example if a custom class is needed.>
            suite.add_expectation(<Code generated by you>)
            batch.validate(suite)
            You should also consider that all values are of the type string. Even those look numeric, e.g. plz!
            Preferabel use gx.expectations that exists for great expectation 1.9.1. Name and description are not valid parameter. Here is a list of those:
            {getExpectationLi(expectationLiId)}
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
    elif promptId == 11:
        prompt = dedent(f"""
            Hi I have the following Data with an error. Generate Great Expectation expectations that flag the data with the errors as invalid. 
            I am using Version 1.9.1.
            The expectations generated by you will be part of a suite of already existing expectations that checks a larger data.frame.
            You do not need to concern yourself with expectations that already exist.
            Data with Error:
            {polluted_md}
            # Information for the Error Detection:
            {err_dect_info}
            Here are some information to help you find the errors: 
            {errLocInfo}
            # Information about the Syntax of 'Great Expectation' Expectations
            ## batch
            This is how I want to use the code you create:
            suite = gx.ExpectationSuite(name="expectation_suite_name")
            <Additional Code run here, for example if a custom class is needed.>
            suite.add_expectation(<Code generated by you>)
            batch.validate(suite)
            You should also consider that all values are of the type string. Even those look numeric, e.g. plz!
            Preferabel use gx.expectations that exists for great expectation 1.9.1. Name and description are not valid parameter. Here is a list of those:
            {getExpectationLi(expectationLiId)}
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
        raise ValueError(
            f"Unsupported promptId {promptId}. Use one of 8, 9, 10, or 11."
        )
    return prompt.strip(), idVals

