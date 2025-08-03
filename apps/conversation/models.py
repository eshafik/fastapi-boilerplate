
# conversation/models.py
from tortoise import fields, models


class ExampleModel(models.Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=255)

    class Meta:
        default_connection = "default"
        table = "conversation_test_example_table"
