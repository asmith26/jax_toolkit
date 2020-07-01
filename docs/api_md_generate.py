import inspect
import re
from typing import Callable

from api_md_CONFIG import accessors, ROOT_GITHUB_URL, PACKAGE_NAME


def get_pretty_function_signature(function: Callable) -> str:
    function_source = inspect.getsource(function)
    function_source_split = function_source.split(" -> ")

    # Clean inputs
    function_name_and_args = function_source_split[0]
    function_name_and_args_split = function_name_and_args.split("self,")
    if len(function_name_and_args_split) != 1:  # if 1, no arguments
        _, arguments_and_type_annotations = function_name_and_args_split
        type_annotations_with_fluff = arguments_and_type_annotations.split(": ")[1:]  # 0 always a function arg (not type)
        last_type_annotation = type_annotations_with_fluff.pop().strip(")").strip().strip(",")  # last one has the function bracket closure
        type_annotations_not_last = ["".join(type_annotation_with_fluff.split(",")[:-1]) for type_annotation_with_fluff in type_annotations_with_fluff]  # want to keep everything before the last comma (in case type uses comma, e.g. Tuple)
        type_annotations = type_annotations_not_last + [last_type_annotation]
        arguments = inspect.getfullargspec(function).args[1:]  # drop self
        input_list = []
        for argument, type_annotation in zip(arguments, type_annotations):
            # Colour arguments green, types blue
            input_list.append(f"<span style='color:green'>{argument}</span>: <span style='color:blue'>{type_annotation}</span>")
        inputs = f'({", ".join(input_list)})'
    else:
        inputs = f'()'

    # Clean outputs
    if len(function_source_split) != 1:  # if 1, function does not return anything (this is intentional to prevent requiring a holoviews dependency)
        return_types_and_fluff = function_source_split[1]
        outputs = return_types_and_fluff.split(":")[0]
        outputs = outputs.replace("pandas.core.frame.DataFrame", "pd.DataFrame")
        outputs = outputs.replace("pandas.core.series.Series", "pd.Series")
        outputs = outputs.replace("jax.numpy.lax_numpy.ndarray", "jnp.ndarray")
    else:
        outputs = "None"

    return f"{inputs} -> {outputs}"


with open("docs/api/accessors.md", "w") as accessors_file:
    accessors_file.writelines("# Accessors API\n")
    accessors_file.writelines("\n")
    for accessor_group, accessors in accessors.items():
        accessors_file.writelines(f"## {accessor_group} Methods\n")
        for accessor in accessors:
            function_name = accessor.__name__
            function_signature = get_pretty_function_signature(accessor)
            docstring = inspect.getdoc(accessor)
            absolute_file_path = inspect.getfile(accessor)
            file_path = re.sub(f".*/{PACKAGE_NAME}/{PACKAGE_NAME}/", "", absolute_file_path)
            github_line_number = inspect.findsource(accessor)[1] + 1
            github_source_url = f"{ROOT_GITHUB_URL}{file_path}#L{github_line_number}"

            accessors_file.writelines(f"### `{function_name}` *<small>[[source]({github_source_url})]</small>*\n")
            accessors_file.writelines(f"`{function_name}`*{function_signature}*\n")
            accessors_file.writelines("\n")
            accessors_file.writelines(f"{docstring}")
            accessors_file.writelines(f"\n\n")
