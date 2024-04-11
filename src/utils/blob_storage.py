from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas
import datetime
import os

AZURE_STORAGE_CONN_STRING = os.environ.get('AZURE_STORAGE_CONN_STRING')
AZURE_STORAGE_CONTAINER = os.environ.get('AZURE_STORAGE_CONTAINER', 'sadtalker')


class BlobStorage:
    def __init__(self):
        self.connection_string = AZURE_STORAGE_CONN_STRING
        self.container_name = AZURE_STORAGE_CONTAINER
        self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        self.blob_client = None

    def upload_file(self, file_path, directory):
        self.blob_client = self.blob_service_client.get_blob_client(self.container_name,
                                                                    f'{directory}/{os.path.basename(file_path)}')

        with open(file_path, "rb") as upload_file:
            self.blob_client.upload_blob(upload_file)

        return self.blob_client

    def get_file_url(self, valid_for_days=30):
        assert self.blob_client is not None, "No file has been uploaded yet."

        sas_token = generate_blob_sas(
            self.blob_client.account_name,
            self.blob_client.container_name,
            self.blob_client.blob_name,
            account_key=self.blob_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.datetime.utcnow() + datetime.timedelta(days=valid_for_days))

        url = f"https://{self.blob_client.account_name}.blob.core.windows.net/{self.blob_client.container_name}/{self.blob_client.blob_name}?{sas_token}"

        return url
