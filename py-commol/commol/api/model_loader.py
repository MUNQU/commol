from pathlib import Path

from commol.context.model import Model


class ModelLoader:
    """
    Loads a compartment model from a file.

    This class provides static methods to load a Model instance from various
    file formats, acting as a factory for Model objects.
    """

    @staticmethod
    def from_json(file_path: str | Path) -> Model:
        """
        Loads a model from a JSON file.

        The method reads the specified JSON file, parses its content, and validates
        it against the Model schema.

        Parameters
        ----------
        file_path : str | Path
            The path to the JSON file.

        Returns
        -------
        Model
            A validated Model instance.

        Raises
        ------
        FileNotFoundError
            If the file at `file_path` does not exist.
        pydantic.ValidationError
            If the JSON content does not conform to the Model schema.
        """
        with open(file_path, "r") as f:
            json_data = f.read()

        return Model.model_validate_json(json_data)
