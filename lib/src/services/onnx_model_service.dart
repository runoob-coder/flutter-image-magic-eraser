import 'dart:async';
import 'dart:developer';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import 'package:path_provider/path_provider.dart';
import 'package:crypto/crypto.dart';

import '../models/model_init_data.dart';
import 'model_download_service.dart';

/// Service for handling ONNX model operations
class OnnxModelService {
  OnnxModelService._internal() {
    // Forward download progress from the download service
    _downloadProgressSubscription = ModelDownloadService
        .instance.downloadProgressStream
        .listen(_forwardDownloadProgress);
  }

  static final OnnxModelService _instance = OnnxModelService._internal();

  /// Returns the singleton instance of the service
  static OnnxModelService get instance => _instance;

  final OnnxRuntime _onnxRuntime = OnnxRuntime();

  /// The ONNX session used for inference
  OrtSession? _session;

  /// Stream controller for broadcasting model loading state changes
  final _stateController = StreamController<ModelLoadingState>.broadcast();

  /// Current state of model loading
  ModelLoadingState _currentState = ModelLoadingState.notLoaded;

  /// Stream of model loading state changes
  Stream<ModelLoadingState> get stateStream => _stateController.stream;

  /// Current state of model loading
  ModelLoadingState get currentState => _currentState;

  /// Stream controller for broadcasting download progress
  final _downloadProgressController =
      StreamController<DownloadProgress>.broadcast();

  /// Stream of download progress updates
  Stream<DownloadProgress> get downloadProgressStream =>
      _downloadProgressController.stream;

  /// Subscription to the download service progress stream
  late StreamSubscription<DownloadProgress> _downloadProgressSubscription;

  /// Directory where downloaded models are stored
  String? _modelStorageDirectory;

  /// Get the directory where models are stored
  Future<String> get _modelDir async {
    if (_modelStorageDirectory != null) return _modelStorageDirectory!;

    final appDir = await getApplicationDocumentsDirectory();
    _modelStorageDirectory = '${appDir.path}/models';

    // Create the directory if it doesn't exist
    final directory = Directory(_modelStorageDirectory!);
    if (!directory.existsSync()) {
      await directory.create(recursive: true);
    }

    return _modelStorageDirectory!;
  }

  /// Updates the current state and broadcasts it to listeners
  void _setState(ModelLoadingState newState) {
    _currentState = newState;
    _stateController.add(newState);
  }

  /// Forwards download progress from the download service to our own stream
  void _forwardDownloadProgress(DownloadProgress progress) {
    _downloadProgressController.add(progress);
  }

  /// Initializes the ONNX environment and creates a session from a URL
  ///
  /// Downloads the model if it doesn't exist locally, then initializes it
  /// Provides progress updates via the downloadProgressStream
  ///
  /// - [modelUrl]: URL to download the model from
  /// - [expectedChecksum]: SHA-256 checksum to verify the downloaded file integrity.
  ///   This is required for security and integrity verification.
  Future<void> initializeModelFromUrl(
    String modelUrl,
    String expectedChecksum,
  ) async {
    try {
      // Use the checksum as the filename for better identification and verification
      final modelPath =
          await _ensureModelDownloaded(modelUrl, expectedChecksum);

      // Add explicit GC hint between file verification and model loading
      // This doesn't force GC but helps suggest it's a good time to run it
      await Future.delayed(Duration.zero);

      // Now initialize from the local file
      await initializeModel(modelPath, isAsset: false);

      if (kDebugMode) {
        log('Model initialized from URL successfully',
            name: "OnnxModelService");
      }
    } catch (e) {
      // The specific error state is already set by the method that threw the exception
      if (kDebugMode) {
        log('Error initializing model from URL: $e',
            name: "OnnxModelService", error: e);
      }
      rethrow;
    }
  }

  /// Static method to calculate checksum using a memory-efficient chunked approach
  static Future<String> _calculateChecksumInIsolateWithChunks(
      String filePath) async {
    final file = File(filePath);
    if (!await file.exists()) {
      throw Exception('File does not exist: $filePath');
    }

    // Create SHA256 hasher that processes chunks
    final digest = await sha256.bind(file.openRead()).first;
    return digest.toString();
  }

  /// Static method for verifying file integrity in an isolate with chunked reading
  static Future<bool> _verifyFileIntegrityInIsolate(
      FileIntegrityData data) async {
    // If no checksum is provided, assume file is valid
    if (data.expectedChecksum == null) {
      return true;
    }

    try {
      final actualChecksum =
          await _calculateChecksumInIsolateWithChunks(data.filePath);
      return actualChecksum.toLowerCase() ==
          data.expectedChecksum!.toLowerCase();
    } catch (e) {
      // Return false on any error
      return false;
    }
  }

  /// Verifies if a file's checksum matches the expected value using chunks to reduce memory usage
  Future<bool> _verifyFileIntegrity(
      String filePath, String? expectedChecksum) async {
    // If no checksum is provided, assume file is valid
    if (expectedChecksum == null) {
      if (kDebugMode) {
        log('No checksum provided for verification, assuming file is valid: $filePath',
            name: "OnnxModelService");
      }
      return true;
    }

    try {
      if (kDebugMode) {
        log('Starting memory-efficient checksum verification for: $filePath',
            name: "OnnxModelService");
      }

      // Run the verification in a background isolate with chunked reading
      final isValid = await compute(_verifyFileIntegrityInIsolate,
          FileIntegrityData(filePath, expectedChecksum));

      if (kDebugMode) {
        if (isValid) {
          log('Checksum verification successful: $filePath',
              name: "OnnxModelService");
        } else {
          log('Checksum verification failed: $filePath',
              name: "OnnxModelService");
          log('Expected: $expectedChecksum', name: "OnnxModelService");
        }
      }

      return isValid;
    } catch (e) {
      if (kDebugMode) {
        log('Error verifying file integrity: $e',
            name: "OnnxModelService", error: e);
      }
      return false;
    }
  }

  /// Checks if the model exists locally, downloads if needed
  Future<String> _ensureModelDownloaded(String url, String checksum) async {
    final modelDir = await _modelDir;
    // Use checksum as the filename with a .onnx extension
    final file = File('$modelDir/$checksum.onnx');

    // Check if the file already exists
    if (await file.exists()) {
      if (kDebugMode) {
        log('Model already downloaded: ${file.path}', name: "OnnxModelService");
      }
      // Verify file integrity with checksum
      final isValid = await _verifyFileIntegrity(file.path, checksum);

      if (!isValid) {
        if (kDebugMode) {
          log('Existing model file is corrupted, re-downloading: ${file.path}',
              name: "OnnxModelService");
        }

        // Delete corrupted file and download again
        await file.delete();
        _setState(ModelLoadingState.downloading);

        try {
          // Use ModelDownloadService to download the file
          await ModelDownloadService.instance.downloadFile(
            url,
            file.path,
            minSize: 1024 * 10, // ONNX models are typically at least 10KB
          );

          // Only verify integrity if download succeeded
          final isNewFileValid =
              await _verifyFileIntegrity(file.path, checksum);
          if (!isNewFileValid) {
            _setState(ModelLoadingState.checksumError);
            throw Exception(
                'Downloaded model file failed integrity check. Expected checksum: $checksum');
          }
        } catch (e) {
          // Set download error state and rethrow
          _setState(ModelLoadingState.downloadError);
          rethrow;
        }
      } else {
        if (kDebugMode) {
          log('Model already downloaded and verified: ${file.path}',
              name: "OnnxModelService");
        }
      }
      return file.path;
    }

    if (kDebugMode) {
      log('Downloading model from $url', name: "OnnxModelService");
    }

    // File doesn't exist, download it
    _setState(ModelLoadingState.downloading);
    try {
      // Use ModelDownloadService to download the file
      await ModelDownloadService.instance.downloadFile(
        url,
        file.path,
        minSize: 1024 * 10, // ONNX models are typically at least 10KB
      );

      // Only verify downloaded file integrity after successful download
      final isValid = await _verifyFileIntegrity(file.path, checksum);
      if (!isValid) {
        // Clean up invalid file
        await file.delete();
        _setState(ModelLoadingState.checksumError);
        throw Exception(
            'Downloaded model file failed integrity check. Expected checksum: $checksum');
      }

      return file.path;
    } catch (e) {
      // Set download error state and clean up any partial file
      _setState(ModelLoadingState.downloadError);
      if (await file.exists()) {
        try {
          await file.delete();
        } catch (_) {
          // Ignore errors when deleting partial files
        }
      }
      rethrow;
    }
  }

  /// Initializes the ONNX runtime and creates a session
  ///
  /// This method should be called once before using the model for inference.
  /// Sessions are created on the main isolate because flutter_onnxruntime
  /// uses platform channels.
  Future<void> initializeModel(String modelPath, {bool isAsset = true}) async {
    try {
      _setState(ModelLoadingState.loading);

      if (kDebugMode) {
        log('Loading model from ${isAsset ? "asset" : "file"}: $modelPath',
            name: "OnnxModelService");
      }

      final sessionOptions = OrtSessionOptions(intraOpNumThreads: 2);

      if (isAsset) {
        if (kDebugMode) {
          log('Creating ONNX session from asset', name: "OnnxModelService");
        }
        _session = await _onnxRuntime.createSessionFromAsset(
          modelPath,
          options: sessionOptions,
        );
      } else {
        final file = File(modelPath);
        if (!await file.exists()) {
          _setState(ModelLoadingState.loadingError);
          throw Exception('Model file not found: $modelPath');
        }

        final fileSize = await file.length();
        if (kDebugMode) {
          final sizeMB = (fileSize / (1024 * 1024)).toStringAsFixed(2);
          log('Creating ONNX session from file ($sizeMB MB)',
              name: "OnnxModelService");
        }

        _session = await _onnxRuntime.createSession(
          modelPath,
          options: sessionOptions,
        );
      }

      _setState(ModelLoadingState.loaded);

      await Future.delayed(Duration.zero);

      if (kDebugMode) {
        log('ONNX session created successfully.', name: "OnnxModelService");
      }
    } catch (e) {
      _setState(ModelLoadingState.loadingError);
      if (kDebugMode) {
        log('Error initializing ONNX model: $e',
            name: "OnnxModelService", error: e);
      }
      rethrow;
    }
  }

  /// Runs inference on the model with the given inputs
  ///
  /// [inputs] is a map of input names to [OrtValue] tensors
  /// Returns a map of output names to [OrtValue] tensors
  Future<Map<String, OrtValue>> runInference(
      Map<String, OrtValue> inputs) async {
    if (_session == null) {
      throw Exception("ONNX session not initialized");
    }

    try {
      return await _session!.run(inputs);
    } catch (e) {
      if (kDebugMode) {
        log('Error running inference: $e', name: "OnnxModelService", error: e);
      }
      rethrow;
    }
  }

  /// Checks if the model is loaded
  bool isModelLoaded() {
    return _session != null;
  }

  /// Disposes of resources when the service is no longer needed
  void dispose() {
    final session = _session;
    _session = null;

    if (session != null) {
      unawaited(session.close());
      _setState(ModelLoadingState.notLoaded);
      if (kDebugMode) {
        log('ONNX session released.', name: "OnnxModelService");
      }
    }

    // Cancel the download progress subscription
    _downloadProgressSubscription.cancel();

    // Close the stream controllers
    _stateController.close();
    _downloadProgressController.close();
  }
}
