"""
Google Drive API — Python setup and helper

Supports two authentication methods:
  - Service Account  (server-to-server, no browser needed)
  - OAuth2 / User Account  (for accessing a real user's Drive)

Common operations:
  list_files, get_file, upload_file, update_file, download_file,
  create_folder, move_file, copy_file, delete_file,
  share_file, list_permissions, remove_permission, get_storage_quota

Install dependencies:
  pip install google-api-python-client google-auth google-auth-oauthlib
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO, Literal

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

AuthMethod = Literal["service_account", "oauth2"]


@dataclass
class DriveSettings:
    """
    All configuration for the Google Drive connection lives here.

    Parameters
    ----------
    auth_method:
        "service_account" — use a service-account JSON key file (no browser)
        "oauth2"          — use OAuth2 for a real user's Drive

    credentials_file:
        Path to your credentials file:
        - service_account: the downloaded JSON key file
        - oauth2: the client_secret_*.json downloaded from Google Cloud Console

    token_file:
        (OAuth2 only) Where to persist the user token so you only
        authorize once. Defaults to "token.json" next to the credentials file.

    scopes:
        Drive API scopes. Default gives full read/write access.
        Use ["https://www.googleapis.com/auth/drive.readonly"] for read-only.

    impersonate:
        (Service account only) A GSuite user email to impersonate via
        domain-wide delegation, e.g. "user@yourdomain.com".

    extra:
        Any additional kwargs forwarded to the underlying google-auth calls.
    """
    auth_method: AuthMethod = "service_account"
    credentials_file: str = "service_account.json"
    token_file: str = ""
    scopes: list[str] = field(default_factory=lambda: [
        "https://www.googleapis.com/auth/drive",
    ])
    impersonate: str = ""
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _build_service_account_creds(settings: DriveSettings):
    from google.oauth2 import service_account

    creds = service_account.Credentials.from_service_account_file(
        settings.credentials_file,
        scopes=settings.scopes,
        **settings.extra,
    )
    if settings.impersonate:
        creds = creds.with_subject(settings.impersonate)
    return creds


def _build_oauth2_creds(settings: DriveSettings):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    token_file = settings.token_file or str(
        Path(settings.credentials_file).with_name("token.json")
    )

    creds = None
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, settings.scopes)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                settings.credentials_file, settings.scopes
            )
            creds = flow.run_local_server(port=0)
        with open(token_file, "w") as fh:
            fh.write(creds.to_json())

    return creds


def build_service(settings: DriveSettings):
    """Build and return an authenticated Google Drive service object."""
    try:
        from googleapiclient.discovery import build
    except ImportError:
        raise ImportError("pip install google-api-python-client google-auth google-auth-oauthlib")

    if settings.auth_method == "service_account":
        creds = _build_service_account_creds(settings)
    elif settings.auth_method == "oauth2":
        creds = _build_oauth2_creds(settings)
    else:
        raise ValueError(f"Unknown auth_method: '{settings.auth_method}'")

    return build("drive", "v3", credentials=creds)


# ---------------------------------------------------------------------------
# GoogleDriveNode
# ---------------------------------------------------------------------------

# Fields returned by default on file list/get calls
_DEFAULT_FIELDS = "id, name, mimeType, size, modifiedTime, parents, webViewLink"


class GoogleDriveNode:
    """
    High-level Google Drive helper node.

    Usage
    -----
    settings = DriveSettings(
        auth_method="service_account",
        credentials_file="service_account.json",
    )
    drive = GoogleDriveNode(settings)
    files = drive.list_files(folder_id="root", query="name contains 'report'")
    """

    def __init__(self, settings: DriveSettings) -> None:
        self.settings = settings
        self._service = None  # lazily built

    @property
    def service(self):
        if self._service is None:
            self._service = build_service(self.settings)
        return self._service

    def reconnect(self) -> None:
        """Force re-authentication (e.g. after token expiry)."""
        self._service = None

    # ------------------------------------------------------------------
    # Files — read
    # ------------------------------------------------------------------

    def list_files(
        self,
        *,
        folder_id: str = "root",
        query: str = "",
        page_size: int = 100,
        fields: str = _DEFAULT_FIELDS,
        include_trashed: bool = False,
    ) -> list[dict]:
        """
        List files in a folder.

        Parameters
        ----------
        folder_id:
            The Drive folder ID to list. "root" for the user's root.
        query:
            Additional Drive query string (q parameter), e.g.
            "name contains 'invoice'" or "mimeType = 'application/pdf'".
        page_size:
            Max items per page (Drive fetches up to 1000).
        include_trashed:
            Include trashed files (default False).
        """
        base_q = f"'{folder_id}' in parents"
        if not include_trashed:
            base_q += " and trashed = false"
        if query:
            base_q += f" and ({query})"

        results: list[dict] = []
        page_token = None

        while True:
            resp = (
                self.service.files()
                .list(
                    q=base_q,
                    pageSize=page_size,
                    fields=f"nextPageToken, files({fields})",
                    pageToken=page_token,
                )
                .execute()
            )
            results.extend(resp.get("files", []))
            page_token = resp.get("nextPageToken")
            if not page_token:
                break

        return results

    def get_file(self, file_id: str, fields: str = _DEFAULT_FIELDS) -> dict:
        """Return metadata for a single file."""
        return self.service.files().get(fileId=file_id, fields=fields).execute()

    def search(self, query: str, page_size: int = 50, fields: str = _DEFAULT_FIELDS) -> list[dict]:
        """Full-Drive search using a raw Drive query string."""
        resp = (
            self.service.files()
            .list(q=query, pageSize=page_size, fields=f"files({fields})")
            .execute()
        )
        return resp.get("files", [])

    # ------------------------------------------------------------------
    # Files — write
    # ------------------------------------------------------------------

    def upload_file(
        self,
        *,
        name: str,
        content: bytes | BinaryIO,
        mime_type: str = "application/octet-stream",
        folder_id: str | None = None,
        fields: str = _DEFAULT_FIELDS,
    ) -> dict:
        """
        Upload a new file to Drive.

        Parameters
        ----------
        name:
            File name shown in Drive.
        content:
            Raw bytes or a file-like object.
        mime_type:
            MIME type, e.g. "text/plain", "application/pdf".
        folder_id:
            Parent folder ID. Omit to place in root.
        """
        try:
            from googleapiclient.http import MediaIoBaseUpload
        except ImportError:
            raise ImportError("pip install google-api-python-client")

        metadata: dict = {"name": name}
        if folder_id:
            metadata["parents"] = [folder_id]

        if isinstance(content, (bytes, bytearray)):
            content = io.BytesIO(content)

        media = MediaIoBaseUpload(content, mimetype=mime_type, resumable=True)
        return (
            self.service.files()
            .create(body=metadata, media_body=media, fields=fields)
            .execute()
        )

    def update_file(
        self,
        file_id: str,
        *,
        content: bytes | BinaryIO | None = None,
        mime_type: str = "application/octet-stream",
        new_name: str | None = None,
        fields: str = _DEFAULT_FIELDS,
    ) -> dict:
        """Update an existing file's content and/or name."""
        try:
            from googleapiclient.http import MediaIoBaseUpload
        except ImportError:
            raise ImportError("pip install google-api-python-client")

        metadata: dict = {}
        if new_name:
            metadata["name"] = new_name

        media = None
        if content is not None:
            if isinstance(content, (bytes, bytearray)):
                content = io.BytesIO(content)
            media = MediaIoBaseUpload(content, mimetype=mime_type, resumable=True)

        return (
            self.service.files()
            .update(fileId=file_id, body=metadata, media_body=media, fields=fields)
            .execute()
        )

    def download_file(self, file_id: str) -> bytes:
        """
        Download a file's raw bytes.

        For Google Workspace files (Docs, Sheets …) use export_file() instead.
        """
        try:
            from googleapiclient.http import MediaIoBaseDownload
        except ImportError:
            raise ImportError("pip install google-api-python-client")

        request = self.service.files().get_media(fileId=file_id)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return buf.getvalue()

    def export_file(self, file_id: str, mime_type: str = "text/plain") -> bytes:
        """
        Export a Google Workspace file (Doc, Sheet, Slide…) to another format.

        Common mime_type values:
          "text/plain", "application/pdf",
          "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        """
        data = self.service.files().export(fileId=file_id, mimeType=mime_type).execute()
        return data if isinstance(data, bytes) else data.encode()

    def create_folder(
        self,
        name: str,
        parent_id: str | None = None,
        fields: str = _DEFAULT_FIELDS,
    ) -> dict:
        """Create a folder and return its metadata."""
        metadata: dict = {
            "name": name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parent_id:
            metadata["parents"] = [parent_id]
        return self.service.files().create(body=metadata, fields=fields).execute()

    def move_file(self, file_id: str, new_folder_id: str, fields: str = _DEFAULT_FIELDS) -> dict:
        """Move a file to a different folder."""
        file_meta = self.service.files().get(fileId=file_id, fields="parents").execute()
        previous_parents = ",".join(file_meta.get("parents", []))
        return (
            self.service.files()
            .update(
                fileId=file_id,
                addParents=new_folder_id,
                removeParents=previous_parents,
                fields=fields,
            )
            .execute()
        )

    def copy_file(self, file_id: str, new_name: str | None = None, fields: str = _DEFAULT_FIELDS) -> dict:
        """Copy a file, optionally giving it a new name."""
        body = {}
        if new_name:
            body["name"] = new_name
        return self.service.files().copy(fileId=file_id, body=body, fields=fields).execute()

    def delete_file(self, file_id: str) -> None:
        """Permanently delete a file (bypass Trash)."""
        self.service.files().delete(fileId=file_id).execute()

    def trash_file(self, file_id: str, fields: str = _DEFAULT_FIELDS) -> dict:
        """Move a file to Trash (recoverable)."""
        return self.service.files().update(fileId=file_id, body={"trashed": True}, fields=fields).execute()

    # ------------------------------------------------------------------
    # Sharing / permissions
    # ------------------------------------------------------------------

    def share_file(
        self,
        file_id: str,
        *,
        email: str | None = None,
        role: str = "reader",
        type: str = "user",
        send_notification: bool = False,
    ) -> dict:
        """
        Share a file with a user, group, domain, or anyone.

        Parameters
        ----------
        email:
            Recipient email (required when type is "user" or "group").
        role:
            "reader" | "commenter" | "writer" | "owner"
        type:
            "user" | "group" | "domain" | "anyone"
        """
        body: dict = {"role": role, "type": type}
        if email:
            body["emailAddress"] = email

        return (
            self.service.permissions()
            .create(
                fileId=file_id,
                body=body,
                sendNotificationEmail=send_notification,
                fields="id, role, type, emailAddress",
            )
            .execute()
        )

    def list_permissions(self, file_id: str) -> list[dict]:
        """Return all permissions on a file."""
        resp = (
            self.service.permissions()
            .list(fileId=file_id, fields="permissions(id, role, type, emailAddress)")
            .execute()
        )
        return resp.get("permissions", [])

    def remove_permission(self, file_id: str, permission_id: str) -> None:
        """Remove a permission from a file."""
        self.service.permissions().delete(fileId=file_id, permissionId=permission_id).execute()

    # ------------------------------------------------------------------
    # Account info
    # ------------------------------------------------------------------

    def get_storage_quota(self) -> dict:
        """Return storage usage and limit in bytes."""
        resp = self.service.about().get(fields="storageQuota").execute()
        return resp.get("storageQuota", {})

    def __repr__(self) -> str:
        return (
            f"GoogleDriveNode(auth={self.settings.auth_method}, "
            f"credentials={self.settings.credentials_file})"
        )


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Service-account setup
    sa_settings = DriveSettings(
        auth_method="service_account",
        credentials_file="service_account.json",   # <-- your key file
        scopes=["https://www.googleapis.com/auth/drive"],
    )

    # OAuth2 setup (opens browser on first run, saves token.json)
    oauth_settings = DriveSettings(
        auth_method="oauth2",
        credentials_file="client_secret.json",     # <-- downloaded from GCP Console
        token_file="token.json",
        scopes=["https://www.googleapis.com/auth/drive"],
    )

    print("Service-account node:", GoogleDriveNode(sa_settings))
    print("OAuth2 node         :", GoogleDriveNode(oauth_settings))

    print("\nExamples (requires valid credentials):")
    print("  drive = GoogleDriveNode(sa_settings)")
    print("  files = drive.list_files(folder_id='root')")
    print("  drive.upload_file(name='hello.txt', content=b'Hello!', mime_type='text/plain')")
    print("  data  = drive.download_file(file_id='<id>')")
    print("  drive.share_file(file_id='<id>', email='bob@example.com', role='writer')")
