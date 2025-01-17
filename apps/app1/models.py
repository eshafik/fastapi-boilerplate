from tortoise import fields, models


class ExampleModel(models.Model):
    id = fields.IntField(pk=True)
    name = fields.CharField(max_length=255)

    class Meta:
        default_connection = "default"
        table = "test_example_table"
