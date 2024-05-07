from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas
import datetime
import os

AZURE_STORAGE_CONN_STRING = os.environ.get("AZURE_STORAGE_CONN_STRING")
AZURE_STORAGE_CONTAINER = os.environ.get("AZURE_STORAGE_CONTAINER", "sadtalker")


class BlobStorage:
    def __init__(self):
        self.connection_string = AZURE_STORAGE_CONN_STRING
        self.container_name = AZURE_STORAGE_CONTAINER
        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.connection_string
        )
        self.container_client = self.blob_service_client.get_container_client(
            self.container_name
        )
        self.blob_client = None

    def check_dir_exists(self, dir_name):
        blob_list = self.container_client.list_blobs(name_starts_with=dir_name)

        for _ in blob_list:
            return True
        return False

    def check_blob_exists(self, blob_name):
        try:
            self.container_client.get_blob_client(blob_name).get_blob_properties()
            return True
        except:
            return False

    def upload_file(self, file_path, directory, custom_name=None):
        self.blob_client = self.blob_service_client.get_blob_client(
            self.container_name,
            f"{directory}/{custom_name if custom_name else os.path.basename(file_path)}",
        )

        with open(file_path, "rb") as upload_file:
            self.blob_client.upload_blob(upload_file, overwrite=True)

        return self.blob_client

    def download_file(self, filename, tmpfile):
        blob_client = self.container_client.get_blob_client(filename)
        download_stream = blob_client.download_blob()
        with tmpfile as file:
            file.write(download_stream.readall())
        return tmpfile.name

    def get_file_url(self, valid_for_days=30):
        assert self.blob_client is not None, "No file has been uploaded yet."

        sas_token = generate_blob_sas(
            self.blob_client.account_name,
            self.blob_client.container_name,
            self.blob_client.blob_name,
            account_key=self.blob_client.credential.account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.datetime.utcnow() + datetime.timedelta(days=valid_for_days),
        )

        url = f"https://{self.blob_client.account_name}.blob.core.windows.net/{self.blob_client.container_name}/{self.blob_client.blob_name}?{sas_token}"

        return url
