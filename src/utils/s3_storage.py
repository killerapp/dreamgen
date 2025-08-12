"""S3-compatible storage backend for generated images."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError, NoCredentialsError
from PIL import Image


class S3StorageManager:
    """Manages image storage in S3-compatible backends."""

    def __init__(
        self,
        bucket_name: str,
        endpoint_url: Optional[str] = None,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        region: str = "us-east-1",
        local_cache_dir: str = "./cache",
        public_read: bool = False
    ):
        """Initialize S3 storage manager.
        
        Args:
            bucket_name: S3 bucket name
            endpoint_url: Custom S3 endpoint (for MinIO, R2, etc)
            access_key: AWS access key ID
            secret_key: AWS secret access key
            region: AWS region
            local_cache_dir: Local directory for caching
            public_read: Whether to make uploads publicly readable
        """
        self.bucket_name = bucket_name
        self.endpoint_url = endpoint_url
        self.region = region
        self.public_read = public_read
        self.local_cache_dir = Path(local_cache_dir)
        self.local_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure S3 client
        self.s3_client = self._create_s3_client(
            endpoint_url, access_key, secret_key, region
        )
        
        # Ensure bucket exists
        self._ensure_bucket_exists()
        
        # Cache for metadata
        self.metadata_cache: Dict[str, Any] = {}
    
    def _create_s3_client(
        self,
        endpoint_url: Optional[str],
        access_key: Optional[str],
        secret_key: Optional[str],
        region: str
    ) -> boto3.client:
        """Create S3 client with provided credentials.
        
        Args:
            endpoint_url: Custom S3 endpoint
            access_key: Access key ID
            secret_key: Secret access key
            region: AWS region
            
        Returns:
            Configured boto3 S3 client
        """
        config = BotoConfig(
            region_name=region,
            signature_version='s3v4',
            retries={'max_attempts': 3, 'mode': 'standard'}
        )
        
        # Build client arguments
        client_args = {
            'service_name': 's3',
            'config': config
        }
        
        if endpoint_url:
            client_args['endpoint_url'] = endpoint_url
        
        if access_key and secret_key:
            client_args['aws_access_key_id'] = access_key
            client_args['aws_secret_access_key'] = secret_key
        
        return boto3.client(**client_args)
    
    def _ensure_bucket_exists(self):
        """Ensure the S3 bucket exists, create if not."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # Bucket doesn't exist, create it
                try:
                    if self.region == 'us-east-1':
                        self.s3_client.create_bucket(Bucket=self.bucket_name)
                    else:
                        self.s3_client.create_bucket(
                            Bucket=self.bucket_name,
                            CreateBucketConfiguration={'LocationConstraint': self.region}
                        )
                    print(f"Created bucket: {self.bucket_name}")
                    
                    # Set bucket policy for public read if requested
                    if self.public_read:
                        self._set_public_read_policy()
                except Exception as create_error:
                    print(f"Failed to create bucket: {create_error}")
                    raise
            else:
                raise
    
    def _set_public_read_policy(self):
        """Set bucket policy to allow public read access."""
        policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "PublicReadGetObject",
                    "Effect": "Allow",
                    "Principal": "*",
                    "Action": "s3:GetObject",
                    "Resource": f"arn:aws:s3:::{self.bucket_name}/*"
                }
            ]
        }
        
        self.s3_client.put_bucket_policy(
            Bucket=self.bucket_name,
            Policy=json.dumps(policy)
        )
    
    def _generate_s3_key(self, prompt: str, timestamp: Optional[datetime] = None) -> str:
        """Generate S3 key for an image based on prompt and timestamp.
        
        Args:
            prompt: Image generation prompt
            timestamp: Generation timestamp
            
        Returns:
            S3 object key
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Create hash of prompt for uniqueness
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()[:8]
        
        # Organize by year/month/day
        date_path = timestamp.strftime("%Y/%m/%d")
        
        # Generate filename
        filename = f"{timestamp.strftime('%H%M%S')}_{prompt_hash}.png"
        
        return f"images/{date_path}/{filename}"
    
    async def upload_image(
        self,
        image_path: Path,
        prompt: str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ) -> Tuple[str, str]:
        """Upload an image to S3.
        
        Args:
            image_path: Local path to image file
            prompt: Generation prompt
            metadata: Additional metadata to store
            timestamp: Generation timestamp
            
        Returns:
            Tuple of (S3 key, public URL if available)
        """
        # Generate S3 key
        s3_key = self._generate_s3_key(prompt, timestamp)
        
        # Prepare metadata
        full_metadata = {
            'prompt': prompt,
            'timestamp': (timestamp or datetime.now()).isoformat(),
            'original_filename': image_path.name
        }
        if metadata:
            full_metadata.update(metadata)
        
        # Convert metadata to S3 format (must be strings)
        s3_metadata = {k: str(v) for k, v in full_metadata.items()}
        
        try:
            # Upload file with metadata
            with open(image_path, 'rb') as f:
                extra_args = {
                    'Metadata': s3_metadata,
                    'ContentType': 'image/png'
                }
                
                if self.public_read:
                    extra_args['ACL'] = 'public-read'
                
                self.s3_client.upload_fileobj(
                    f,
                    self.bucket_name,
                    s3_key,
                    ExtraArgs=extra_args
                )
            
            # Store prompt text file
            prompt_key = s3_key.replace('.png', '.txt')
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=prompt_key,
                Body=prompt.encode('utf-8'),
                ContentType='text/plain'
            )
            
            # Cache metadata
            self.metadata_cache[s3_key] = full_metadata
            
            # Generate public URL if applicable
            public_url = ""
            if self.public_read:
                if self.endpoint_url:
                    public_url = f"{self.endpoint_url}/{self.bucket_name}/{s3_key}"
                else:
                    public_url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
            
            return s3_key, public_url
            
        except Exception as e:
            print(f"Failed to upload to S3: {e}")
            raise
    
    async def download_image(self, s3_key: str, local_path: Optional[Path] = None) -> Path:
        """Download an image from S3.
        
        Args:
            s3_key: S3 object key
            local_path: Optional local path to save to
            
        Returns:
            Path to downloaded file
        """
        if local_path is None:
            # Use cache directory
            local_path = self.local_cache_dir / s3_key.split('/')[-1]
        
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.s3_client.download_file(
                self.bucket_name,
                s3_key,
                str(local_path)
            )
            return local_path
        except Exception as e:
            print(f"Failed to download from S3: {e}")
            raise
    
    def list_images(
        self,
        prefix: Optional[str] = None,
        max_results: int = 100,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """List images in S3 bucket.
        
        Args:
            prefix: S3 key prefix to filter by
            max_results: Maximum number of results
            start_date: Filter images after this date
            end_date: Filter images before this date
            
        Returns:
            List of image metadata dictionaries
        """
        images = []
        
        # Build prefix for date filtering if provided
        if start_date and not prefix:
            prefix = f"images/{start_date.strftime('%Y/%m')}"
        elif not prefix:
            prefix = "images/"
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=prefix,
                PaginationConfig={'MaxItems': max_results * 2}  # Get extra for filtering
            )
            
            for page in page_iterator:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    # Skip non-image files
                    if not obj['Key'].endswith('.png'):
                        continue
                    
                    # Apply date filtering
                    if start_date and obj['LastModified'].replace(tzinfo=None) < start_date:
                        continue
                    if end_date and obj['LastModified'].replace(tzinfo=None) > end_date:
                        continue
                    
                    # Get metadata
                    metadata = self._get_image_metadata(obj['Key'])
                    
                    images.append({
                        'key': obj['Key'],
                        'size': obj['Size'],
                        'last_modified': obj['LastModified'].isoformat(),
                        'metadata': metadata,
                        'url': self._get_public_url(obj['Key']) if self.public_read else None
                    })
                    
                    if len(images) >= max_results:
                        return images
            
        except Exception as e:
            print(f"Failed to list images: {e}")
            raise
        
        return images
    
    def _get_image_metadata(self, s3_key: str) -> Dict[str, Any]:
        """Get metadata for an image.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Image metadata dictionary
        """
        # Check cache first
        if s3_key in self.metadata_cache:
            return self.metadata_cache[s3_key]
        
        try:
            # Get object metadata
            response = self.s3_client.head_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            metadata = response.get('Metadata', {})
            
            # Try to get prompt from text file
            if 'prompt' not in metadata:
                prompt_key = s3_key.replace('.png', '.txt')
                try:
                    prompt_obj = self.s3_client.get_object(
                        Bucket=self.bucket_name,
                        Key=prompt_key
                    )
                    metadata['prompt'] = prompt_obj['Body'].read().decode('utf-8')
                except:
                    metadata['prompt'] = 'Unknown'
            
            # Cache and return
            self.metadata_cache[s3_key] = metadata
            return metadata
            
        except Exception:
            return {'prompt': 'Unknown'}
    
    def _get_public_url(self, s3_key: str) -> str:
        """Get public URL for an S3 object.
        
        Args:
            s3_key: S3 object key
            
        Returns:
            Public URL string
        """
        if self.endpoint_url:
            return f"{self.endpoint_url}/{self.bucket_name}/{s3_key}"
        else:
            return f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{s3_key}"
    
    def get_presigned_url(self, s3_key: str, expiration: int = 3600) -> str:
        """Generate a presigned URL for temporary access.
        
        Args:
            s3_key: S3 object key
            expiration: URL expiration time in seconds
            
        Returns:
            Presigned URL string
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': self.bucket_name, 'Key': s3_key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            print(f"Failed to generate presigned URL: {e}")
            raise
    
    def delete_image(self, s3_key: str):
        """Delete an image from S3.
        
        Args:
            s3_key: S3 object key to delete
        """
        try:
            # Delete image
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=s3_key
            )
            
            # Delete associated prompt file
            prompt_key = s3_key.replace('.png', '.txt')
            self.s3_client.delete_object(
                Bucket=self.bucket_name,
                Key=prompt_key
            )
            
            # Remove from cache
            if s3_key in self.metadata_cache:
                del self.metadata_cache[s3_key]
                
        except Exception as e:
            print(f"Failed to delete from S3: {e}")
            raise
    
    def sync_to_local(self, prefix: str = "images/", local_dir: Optional[Path] = None):
        """Sync S3 images to local directory.
        
        Args:
            prefix: S3 prefix to sync
            local_dir: Local directory to sync to
        """
        if local_dir is None:
            local_dir = self.local_cache_dir
        
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            paginator = self.s3_client.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            for page in page_iterator:
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    # Calculate local path
                    relative_path = obj['Key'].replace(prefix, '', 1)
                    local_path = local_dir / relative_path
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Download if not exists or different size
                    if not local_path.exists() or local_path.stat().st_size != obj['Size']:
                        print(f"Syncing: {obj['Key']}")
                        self.s3_client.download_file(
                            self.bucket_name,
                            obj['Key'],
                            str(local_path)
                        )
                        
        except Exception as e:
            print(f"Failed to sync from S3: {e}")
            raise