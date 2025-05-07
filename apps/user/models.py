from tortoise import fields, models


class User(models.Model):
    id = fields.BigIntField(pk=True)
    username = fields.CharField(unique=True, max_length=45)
    email = fields.CharField(max_length=255, null=True, unique=True)
    password = fields.CharField(max_length=255, null=True)  # hashed password
    name = fields.CharField(max_length=100, null=True)
    joined_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)
    last_login = fields.DatetimeField(null=True)

    class Meta:
        default_connection = "default"
        table = "users"

    def __str__(self):
        return self.email or f"User({self.id})"
