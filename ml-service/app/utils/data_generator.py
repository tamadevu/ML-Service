import random
from faker import Faker
from schemas.data import DataGenerationSchema, LabelledDataSchema

faker = Faker()

class DataGenerator:
    def __init__(self, data_schema: DataGenerationSchema):
        """
        Initializes a new instance of the class with the provided data schema.

        Args:
            data_schema (DataGenerationSchema): The data schema to be used for data generation.

        Returns:
            None
        """
        self.schema = data_schema.schema
        self.number = data_schema.number


    def generate_random_data(self) -> LabelledDataSchema:
        """
        Generate random data based on the given schema.

        Args:
            schema (DataGenerationSchema): The schema used to generate the random data.

        Returns:
            LabelledDataSchema: The generated random data.

        """
        random_data = {
            "name": faker.name(),
            "age": random.randint(self.schema.age.range['start'], self.schema.age.range['end']),
            "income": round(random.uniform(0, 1000000), 2)
        }
        return LabelledDataSchema(**random_data)

    def data_generator(self) -> list[LabelledDataSchema]:
        """
        Generate a list of labelled data based on the given schema and total count.

        Args:
            schema (DataGenerationSchema): The schema used to generate the random data.
            total_count (int): The total number of labelled data to generate.

        Returns:
            list[LabelledDataSchema]: A list of labelled data generated based on the given schema and total count.
        """
        generated_data: list[LabelledDataSchema] = []

        for _ in range(self.number):
            generated_data.append(self.generate_random_data())
        return generated_data
        