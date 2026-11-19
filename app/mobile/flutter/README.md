# VisionMatch AI — Flutter Mobile App

This directory contains the Flutter client for the VisionMatch AI FastAPI service.

## Features

- Text-to-Image retrieval using a natural-language query.
- Image-to-Text retrieval using a photo selected from the device gallery.
- Backend readiness indicator.
- Top-K selection (5, 10, 20, or 50).
- Ranked result cards with similarity scores and image previews.
- Configurable API base URL in the app.

The mobile app is an inference client. It does not train models or load TensorFlow checkpoints; the existing FastAPI backend does that.

## Requirements

- Flutter 3.19+ (Dart 3.3+).
- Android Studio / Android SDK for Android builds.
- Xcode for iOS builds (macOS only).
- A running VisionMatch FastAPI backend with a compatible checkpoint and retrieval gallery.

## Generate native platform scaffolding

The repository keeps the Flutter source and package manifest. From this directory, generate the standard Android and iOS runner projects:

```bash
flutter create --platforms=android,ios .
flutter pub get
```

If you only target one platform, use `--platforms=android` or `--platforms=ios`.

## Run on Android

Start the backend on the development computer so it listens on all interfaces:

```bash
python -m uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
```

Then connect the phone and run:

```bash
flutter run
```

API address:
- Android emulator: `http://10.0.2.2:8000`.
- Physical Android phone: use the computer's LAN IPv4 address, for example `http://192.168.1.10:8000`. The phone and computer must be on the same network.
- Production: use the deployed HTTPS API URL.

The API URL can be edited in the app's Backend API address field. For local HTTP development, edit the generated `android/app/src/main/AndroidManifest.xml` and add `<uses-permission android:name="android.permission.INTERNET" />` directly under the `<manifest>` element. Set `android:usesCleartextTraffic="true"` on the `<application>` element for local HTTP only. Prefer HTTPS and disable cleartext traffic for production. Do not expose an unauthenticated development API directly to the public internet.

## Run on iOS

Generate the iOS runner on macOS. In `ios/Runner/Info.plist`, add an `NSPhotoLibraryUsageDescription` string explaining that the app uses photo-library access to select an image for retrieval. Configure the API URL to the backend's reachable LAN or HTTPS address, then run:

```bash
flutter run
```

## Dependencies

Flutter dependencies are managed by `pubspec.yaml` and installed with `flutter pub get`. They are intentionally separate from the repository-root Python `requirements.txt`.

## API contract

- `GET /health`: backend and model readiness.
- `POST /search/text`: JSON body containing `query` and `top_k`.
- `POST /search/image?top_k=N`: multipart upload using the `file` field.
- `GET /images/{image_name}`: gallery image preview.

See [Backend API documentation](../../backend/README.md) for configuration and response details.
