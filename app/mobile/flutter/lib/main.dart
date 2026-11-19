import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

void main() => runApp(const VisionMatchApp());

class VisionMatchApp extends StatelessWidget {
  const VisionMatchApp({super.key});

  @override
  Widget build(BuildContext context) {
    const ink = Color(0xFF111827);
    const violet = Color(0xFF6957E8);
    return MaterialApp(
      title: 'VisionMatch AI',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: ColorScheme.fromSeed(
          seedColor: violet,
          brightness: Brightness.light,
          surface: const Color(0xFFF7F7FC),
        ),
        scaffoldBackgroundColor: const Color(0xFFF7F7FC),
        appBarTheme: const AppBarTheme(
          backgroundColor: Color(0xFFF7F7FC),
          foregroundColor: ink,
          centerTitle: false,
          elevation: 0,
        ),
        inputDecorationTheme: InputDecorationTheme(
          filled: true,
          fillColor: Colors.white,
          border: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: const BorderSide(color: Color(0xFFE6E7EF)),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: const BorderSide(color: Color(0xFFE6E7EF)),
          ),
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(16),
            borderSide: const BorderSide(color: violet, width: 1.5),
          ),
        ),
      ),
      home: const RetrievalHomePage(),
    );
  }
}

enum SearchMode { textToImage, imageToText }

class RetrievalHomePage extends StatefulWidget {
  const RetrievalHomePage({super.key});

  @override
  State<RetrievalHomePage> createState() => _RetrievalHomePageState();
}

class _RetrievalHomePageState extends State<RetrievalHomePage> {
  final _apiController =
      TextEditingController(text: 'http://10.0.2.2:8000');
  final _queryController = TextEditingController();
  final _picker = ImagePicker();

  SearchMode _mode = SearchMode.textToImage;
  XFile? _selectedImage;
  int _topK = 5;
  bool _loading = false;
  bool? _apiReady;
  String? _apiMessage;
  String? _error;
  List<Map<String, dynamic>> _results = [];
  double? _elapsedMs;
  String? _modelName;

  String get _baseUrl => _apiController.text.trim().replaceAll(RegExp(r'/+$'), '');

  @override
  void initState() {
    super.initState();
    _checkHealth();
  }

  @override
  void dispose() {
    _apiController.dispose();
    _queryController.dispose();
    super.dispose();
  }

  Future<void> _checkHealth() async {
    final base = _baseUrl;
    if (base.isEmpty) return;
    try {
      final response = await http
          .get(Uri.parse('$base/health'))
          .timeout(const Duration(seconds: 8));
      final data = jsonDecode(response.body) as Map<String, dynamic>;
      final ready = response.statusCode == 200 &&
          data['status'] == 'ok' &&
          data['model_loaded'] == true;
      if (!mounted) return;
      setState(() {
        _apiReady = ready;
        _apiMessage = ready
            ? 'Backend connected'
            : (data['startup_error']?.toString() ??
                'API is running, but the model/gallery is not ready.');
      });
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _apiReady = false;
        _apiMessage = 'Could not reach the backend. Check the API URL.';
      });
    }
  }

  Future<void> _chooseImage() async {
    try {
      final image = await _picker.pickImage(
        source: ImageSource.gallery,
        imageQuality: 92,
      );
      if (image != null && mounted) {
        setState(() {
          _selectedImage = image;
          _error = null;
          _results = [];
        });
      }
    } catch (error) {
      if (mounted) setState(() => _error = 'Could not open the image picker: $error');
    }
  }

  Future<void> _search() async {
    if (_loading) return;
    if (_mode == SearchMode.textToImage && _queryController.text.trim().isEmpty) {
      setState(() => _error = 'Write a description of the image you are looking for.');
      return;
    }
    if (_mode == SearchMode.imageToText && _selectedImage == null) {
      setState(() => _error = 'Choose an image before searching.');
      return;
    }

    FocusScope.of(context).unfocus();
    setState(() {
      _loading = true;
      _error = null;
      _results = [];
      _elapsedMs = null;
    });

    try {
      http.Response response;
      if (_mode == SearchMode.textToImage) {
        response = await http
            .post(
              Uri.parse('$_baseUrl/search/text'),
              headers: {'Content-Type': 'application/json'},
              body: jsonEncode({
                'query': _queryController.text.trim(),
                'top_k': _topK,
              }),
            )
            .timeout(const Duration(seconds: 90));
      } else {
        final request = http.MultipartRequest(
          'POST',
          Uri.parse('$_baseUrl/search/image?top_k=$_topK'),
        );
        request.files.add(
          await http.MultipartFile.fromPath('file', _selectedImage!.path),
        );
        final streamed = await request.send().timeout(const Duration(seconds: 90));
        response = await http.Response.fromStream(streamed);
      }

      final payload = jsonDecode(response.body);
      if (response.statusCode < 200 || response.statusCode >= 300) {
        final detail = payload is Map
            ? (payload['detail'] ?? payload['message'] ?? response.body)
            : response.body;
        throw Exception(detail.toString());
      }
      final data = payload as Map<String, dynamic>;
      final rawResults = (data['results'] as List? ?? []);
      if (!mounted) return;
      setState(() {
        _results = rawResults
            .whereType<Map>()
            .map((item) => Map<String, dynamic>.from(item))
            .toList();
        _elapsedMs = (data['elapsed_ms'] as num?)?.toDouble();
        _modelName = data['model']?.toString();
      });
    } catch (error) {
      if (mounted) {
        setState(() => _error = error.toString().replaceFirst('Exception: ', ''));
      }
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  String _imageUrl(Map<String, dynamic> result) {
    final path = result['image_url']?.toString();
    if (path == null || path.isEmpty) return '';
    return Uri.parse(_baseUrl).resolve(path).toString();
  }

  @override
  Widget build(BuildContext context) {
    final isText = _mode == SearchMode.textToImage;
    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            Container(
              width: 38,
              height: 38,
              decoration: BoxDecoration(
                color: const Color(0xFFECE9FF),
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Icon(Icons.auto_awesome, color: Color(0xFF6957E8)),
            ),
            const SizedBox(width: 11),
            const Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('VisionMatch AI', style: TextStyle(fontWeight: FontWeight.w800, fontSize: 18)),
                Text('Image & text retrieval', style: TextStyle(fontSize: 11, color: Colors.black54)),
              ],
            ),
          ],
        ),
        actions: [
          IconButton(
            tooltip: 'Check backend connection',
            onPressed: _checkHealth,
            icon: const Icon(Icons.refresh_rounded),
          ),
          const SizedBox(width: 6),
        ],
      ),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.fromLTRB(18, 8, 18, 28),
          children: [
            _connectionCard(),
            const SizedBox(height: 22),
            const Text('Find what you’re looking for',
                style: TextStyle(fontSize: 23, fontWeight: FontWeight.w800, color: Color(0xFF17172A))),
            const SizedBox(height: 6),
            const Text('Search with a sentence or use a photo as your query.',
                style: TextStyle(color: Colors.black54, height: 1.4)),
            const SizedBox(height: 20),
            SegmentedButton<SearchMode>(
              showSelectedIcon: false,
              segments: const [
                ButtonSegment(
                  value: SearchMode.textToImage,
                  icon: Icon(Icons.text_fields_rounded),
                  label: Text('Text → Image'),
                ),
                ButtonSegment(
                  value: SearchMode.imageToText,
                  icon: Icon(Icons.image_search_rounded),
                  label: Text('Image → Text'),
                ),
              ],
              selected: {_mode},
              onSelectionChanged: (selection) => setState(() {
                _mode = selection.first;
                _error = null;
                _results = [];
              }),
            ),
            const SizedBox(height: 16),
            if (isText) _textQueryCard() else _imageQueryCard(),
            const SizedBox(height: 16),
            _topKSelector(),
            const SizedBox(height: 16),
            FilledButton.icon(
              onPressed: _loading ? null : _search,
              icon: _loading
                  ? const SizedBox(
                      width: 18,
                      height: 18,
                      child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                    )
                  : const Icon(Icons.search_rounded),
              label: Text(_loading ? 'Searching…' : 'Search'),
              style: FilledButton.styleFrom(
                minimumSize: const Size.fromHeight(54),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
                textStyle: const TextStyle(fontWeight: FontWeight.w700, fontSize: 16),
              ),
            ),
            if (_error != null) ...[
              const SizedBox(height: 14),
              _messageCard(_error!, isError: true),
            ],
            if (_results.isNotEmpty) ...[
              const SizedBox(height: 26),
              Row(
                children: [
                  const Expanded(
                    child: Text('Search results',
                        style: TextStyle(fontSize: 19, fontWeight: FontWeight.w800)),
                  ),
                  if (_elapsedMs != null)
                    Text('${_elapsedMs!.toStringAsFixed(0)} ms',
                        style: const TextStyle(fontSize: 12, color: Colors.black54)),
                ],
              ),
              if (_modelName != null)
                Padding(
                  padding: const EdgeInsets.only(top: 4, bottom: 12),
                  child: Text('Served by $_modelName',
                      style: const TextStyle(color: Colors.black54, fontSize: 12)),
                ),
              ..._results.map((result) => _resultCard(result, isText)),
            ],
          ],
        ),
      ),
    );
  }

  Widget _connectionCard() {
    final ready = _apiReady == true;
    final pending = _apiReady == null;
    final color = ready ? const Color(0xFF16845B) : const Color(0xFFB66A17);
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: ready ? const Color(0xFFEAF7F1) : const Color(0xFFFFF5E8),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: color.withOpacity(.16)),
      ),
      child: Row(
        children: [
          Icon(ready ? Icons.cloud_done_rounded : Icons.cloud_off_rounded, color: color),
          const SizedBox(width: 10),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(pending ? 'Checking backend…' : (ready ? 'Backend ready' : 'Backend unavailable'),
                    style: TextStyle(fontWeight: FontWeight.w700, color: color)),
                Text(_apiMessage ?? 'Connecting to VisionMatch API',
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(fontSize: 11, color: Colors.black54)),
              ],
            ),
          ),
          IconButton(
            onPressed: _checkHealth,
            icon: const Icon(Icons.sync_rounded),
            color: color,
          ),
        ],
      ),
    );
  }

  Widget _textQueryCard() => Container(
        padding: const EdgeInsets.all(16),
        decoration: _cardDecoration(),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Describe an image',
                style: TextStyle(fontWeight: FontWeight.w700, fontSize: 15)),
            const SizedBox(height: 12),
            TextField(
              controller: _queryController,
              minLines: 3,
              maxLines: 5,
              maxLength: 1000,
              decoration: const InputDecoration(
                hintText: 'e.g. A dog running through a green field…',
                alignLabelWithHint: true,
              ),
            ),
          ],
        ),
      );

  Widget _imageQueryCard() => Container(
        padding: const EdgeInsets.all(16),
        decoration: _cardDecoration(),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Choose a photo',
                style: TextStyle(fontWeight: FontWeight.w700, fontSize: 15)),
            const SizedBox(height: 12),
            if (_selectedImage != null)
              ClipRRect(
                borderRadius: BorderRadius.circular(14),
                child: Image.file(
                  File(_selectedImage!.path),
                  height: 190,
                  width: double.infinity,
                  fit: BoxFit.cover,
                ),
              )
            else
              Container(
                height: 150,
                width: double.infinity,
                decoration: BoxDecoration(
                  color: const Color(0xFFF4F2FF),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: const Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.add_photo_alternate_outlined, size: 38, color: Color(0xFF6957E8)),
                    SizedBox(height: 8),
                    Text('Your selected image will appear here',
                        style: TextStyle(color: Colors.black54, fontSize: 12)),
                  ],
                ),
              ),
            const SizedBox(height: 12),
            OutlinedButton.icon(
              onPressed: _chooseImage,
              icon: const Icon(Icons.photo_library_outlined),
              label: Text(_selectedImage == null ? 'Choose from gallery' : 'Change image'),
              style: OutlinedButton.styleFrom(
                minimumSize: const Size.fromHeight(48),
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(14)),
              ),
            ),
          ],
        ),
      );

  Widget _topKSelector() => Container(
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 13),
        decoration: _cardDecoration(),
        child: Row(
          children: [
            const Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('Number of results', style: TextStyle(fontWeight: FontWeight.w700)),
                  SizedBox(height: 3),
                  Text('Choose how many matches to retrieve',
                      style: TextStyle(fontSize: 11, color: Colors.black54)),
                ],
              ),
            ),
            DropdownButton<int>(
              value: _topK,
              underline: const SizedBox.shrink(),
              items: const [5, 10, 20, 50]
                  .map((value) => DropdownMenuItem(value: value, child: Text('$value')))
                  .toList(),
              onChanged: (value) => setState(() => _topK = value ?? 5),
            ),
          ],
        ),
      );

  Widget _resultCard(Map<String, dynamic> result, bool textMode) {
    final rank = result['rank']?.toString() ?? '—';
    final similarity = (result['similarity'] as num?)?.toDouble() ??
        (result['score'] as num?)?.toDouble() ?? 0;
    final caption = result['caption']?.toString() ??
        (result['result'] is Map ? result['result']['caption']?.toString() : null);
    final imageUrl = _imageUrl(result);
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: _cardDecoration(),
      clipBehavior: Clip.antiAlias,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          if (textMode && imageUrl.isNotEmpty)
            Image.network(
              imageUrl,
              height: 190,
              width: double.infinity,
              fit: BoxFit.cover,
              errorBuilder: (_, __, ___) => Container(
                height: 110,
                color: const Color(0xFFF0EFF5),
                alignment: Alignment.center,
                child: const Icon(Icons.broken_image_outlined, color: Colors.black38, size: 34),
              ),
            ),
          Padding(
            padding: const EdgeInsets.all(14),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 9, vertical: 5),
                      decoration: BoxDecoration(
                        color: const Color(0xFFECE9FF),
                        borderRadius: BorderRadius.circular(20),
                      ),
                      child: Text('RANK $rank',
                          style: const TextStyle(
                              color: Color(0xFF5947D6),
                              fontWeight: FontWeight.w800,
                              fontSize: 10)),
                    ),
                    const Spacer(),
                    Text('Similarity  ${(similarity * 100).toStringAsFixed(1)}%',
                        style: const TextStyle(fontWeight: FontWeight.w700, fontSize: 12)),
                  ],
                ),
                if (!textMode && caption != null) ...[
                  const SizedBox(height: 10),
                  Text(caption, style: const TextStyle(fontSize: 14, height: 1.45)),
                ],
                if (textMode && result['image_id'] != null) ...[
                  const SizedBox(height: 8),
                  Text(result['image_id'].toString(),
                      style: const TextStyle(fontSize: 11, color: Colors.black54)),
                ],
              ],
            ),
          ),
        ],
      ),
    );
  }

  BoxDecoration _cardDecoration() => BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(18),
        border: Border.all(color: const Color(0xFFE9E9F0)),
      );

  Widget _messageCard(String message, {bool isError = false}) => Container(
        padding: const EdgeInsets.all(13),
        decoration: BoxDecoration(
          color: isError ? const Color(0xFFFFEEEE) : const Color(0xFFEAF7F1),
          borderRadius: BorderRadius.circular(14),
        ),
        child: Row(
          children: [
            Icon(isError ? Icons.error_outline : Icons.info_outline,
                color: isError ? Colors.redAccent : Colors.green),
            const SizedBox(width: 9),
            Expanded(child: Text(message, style: const TextStyle(fontSize: 12))),
          ],
        ),
      );
}
