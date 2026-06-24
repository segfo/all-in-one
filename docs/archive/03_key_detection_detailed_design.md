> 📜 **これは現行実装の源流となった初期設計です（歴史記録）。**
> 当初は小節単位の「キー（調性）検出」モジュール（`key_detection/` パッケージ, librosa + music21）を構想していました。
> その中核思想（和声 stem の活用・downbeat = 小節境界での整合）は、現行の madmom ベース「コード検出」
> （stem 重み付き投票 + downbeat 優先スナップ）へと発展しています。
> **現行はコード検出であり、キー（調性）推定は未実装**です。現行の設計と思想は
> [`../02_chord_detection.md`](../02_chord_detection.md) を参照してください。

---

# 小節単位キー検出モジュール - 詳細設計書

## 1. ミキサーモジュール（[`mixer.py`](src/allin1/key_detection/mixer.py)）

### 1.1 クラス設計

```python
class AudioMixer:
    """Demucs 分離音源の再ミックス処理"""
    
    def __init__(self, demix_dir: Path, device: str = 'cpu'):
        """
        Parameters
        ----------
        demix_dir : Path
            Demucs 出力ディレクトリ（例：./demix/htdemucs）
        device : str
            処理デバイス（'cpu', 'cuda', 'mps'）
        """
        self.demix_dir = mkpath(demix_dir)
        self.device = device
    
    def get_demix_path(self, track_name: str) -> Path:
        """
        Demucs 出力ディレクトリを構築する
        
        Parameters
        ----------
        track_name : str
            トラック名（例：'YOASOBI - アイドル'）
        
        Returns
        -------
        Path
            Demucs 出力ディレクトリパス
        """
        return self.demix_dir / 'htdemucs' / track_name
    
    def mix_without_vocals_drums(
        self, 
        track_path: Path,
        cache_dir: Optional[Path] = None
    ) -> Tuple[np.ndarray, int]:
        """
        ボーカル・ドラムを除いた音源をミックスする
        
        処理フロー：
        1. Demucs 分離音源のパスを検出
        2. キャッシュが存在すれば読み込み、なければ再計算
        3. bass.wav と other.wav を加算混合
        4. サンプリングレートを統一して返す
        
        Parameters
        ----------
        track_path : Path
            元音声ファイルのパス
        cache_dir : Optional[Path]
            キャッシュディレクトリ（デフォルト：None）
        
        Returns
        -------
        Tuple[np.ndarray, int]
            (波形データ，サンプリングレート)
        
        Raises
        ------
        AudioLoadError
            Demucs 音源が見つからない場合
        """
        track_name = track_path.stem
        demix_base = self.get_demix_path(track_name)
        
        # キャッシュパスの構築
        if cache_dir is None:
            cache_dir = track_path.parent / '.key_cache'
        mixed_path = cache_dir / f'{track_name}_mixed.wav'
        
        # キャッシュが存在すれば読み込み
        if mixed_path.exists():
            y, sr = librosa.load(str(mixed_path), sr=None, mono=False)
            return y.mean(axis=0), sr  # ステレオの場合はモノラル化
        
        # Demucs 音源の存在確認
        bass_path = demix_base / 'bass.wav'
        other_path = demix_base / 'other.wav'
        
        if not all(p.exists() for p in [bass_path, other_path]):
            raise AudioLoadError(
                f"Demucs 音源が見つかりません：{demix_base}\n"
                "analyze() を実行して音源分離を行ってください。"
            )
        
        # 音源の読み込み（サンプリングレート統一）
        sr = 44100
        y_bass, _ = librosa.load(str(bass_path), sr=sr, mono=False)
        y_other, _ = librosa.load(str(other_path), sr=sr, mono=False)
        
        # ミックス処理（加算）
        y_mixed = y_bass + y_other
        
        # 正規化（クリッピング防止）
        y_mixed = y_mixed / np.max(np.abs(y_mixed)) * 0.95
        
        # キャッシュ保存
        cache_dir.mkdir(parents=True, exist_ok=True)
        save_audio(
            wav=torch.from_numpy(y_mixed),
            path=mixed_path,
            samplerate=sr,
        )
        
        return y_mixed.mean(axis=0), sr
    
    def load_cached_mixed_audio(self, track_name: str) -> Optional[Tuple[np.ndarray, int]]:
        """
        キャッシュされたミックス音源を読み込む
        
        Parameters
        ----------
        track_name : str
            トラック名
        
        Returns
        -------
        Optional[Tuple[np.ndarray, int]]
            (波形データ，サンプリングレート) または None
        """
        cache_dir = self.demix_dir.parent / '.key_cache'
        mixed_path = cache_dir / f'{track_name}_mixed.wav'
        
        if not mixed_path.exists():
            return None
        
        y, sr = librosa.load(str(mixed_path), sr=None, mono=False)
        return y.mean(axis=0), sr
```

---

## 2. 和音検出モジュール（[`chord_detector.py`](src/allin1/key_detection/chord_detector.py)）

### 2.1 クラス設計

```python
@dataclass
class ChordEvent:
    """和音イベント"""
    time: float      # 発音時刻（秒）
    chord: str       # 和音ラベル（例：'C', 'Am', 'G7'）
    confidence: float  # 検出信頼度（0.0～1.0）


class ChordDetector:
    """和音検出モジュール"""
    
    def __init__(self, method: str = 'librosa', sr: int = 44100):
        """
        Parameters
        ----------
        method : str
            検出手法（'librosa' または 'madmom'）
        sr : int
            サンプリングレート
        """
        self.method = method
        self.sr = sr
        
        # librosa のパラメータ設定
        self.hop_length = 512
        self.n_fft = 2048
        self.fmin = librosa.note_to_freq('C1')  # C1=32.70Hz
        self.fmax = librosa.note_to_freq('C7')  # C7=2093.00Hz
    
    def detect_chords(self, y: np.ndarray) -> List[ChordEvent]:
        """
        和音シーケンスを検出する
        
        Parameters
        ----------
        y : np.ndarray
            モノラル波形データ
        
        Returns
        -------
        List[ChordEvent]
            検出した和音イベントのリスト
        """
        if self.method == 'madmom':
            return self._detect_with_madmom(y)
        else:
            return self._detect_with_librosa(y)
    
    def _detect_with_librosa(self, y: np.ndarray) -> List[ChordEvent]:
        """librosa を使用して和音を検出"""
        # チェルニー検出（chroma feature）
        chroma = librosa.feature.chroma_cqt(
            y=y,
            sr=self.sr,
            hop_length=self.hop_length,
            n_chroma=12,
            fmin=self.fmin,
            fmax=self.fmax,
        )
        
        # 時間軸の計算
        frames = librosa.frames_to_time(
            np.arange(chroma.shape[1]),
            sr=self.sr,
            hop_length=self.hop_length,
        )
        
        # クラスタリングによる和音推定（簡易版）
        chords = []
        for i, frame in enumerate(chroma.T):
            time = frames[i]
            
            # 最も強い chroma 成分を抽出
            strongest_notes = np.argsort(frame)[-3:]  # 上位 3 つの音符
            
            # 和音タイプを推定（簡易ルールベース）
            chord_label, confidence = self._infer_chord_type(frame, strongest_notes)
            
            chords.append(ChordEvent(
                time=time,
                chord=chord_label,
                confidence=confidence,
            ))
        
        return chords
    
    def _detect_with_madmom(self, y: np.ndarray) -> List[ChordEvent]:
        """madmom を使用して和音を検出（より高精度）"""
        try:
            from madmom.features.chords import ChordSequenceDetector
            
            detector = ChordSequenceDetector(
                model='cnn14',  # 事前学習済みモデル
                fps=25,         # フレームレート
            )
            
            # 和音シーケンスを検出
            chords = detector(y, self.sr)
            
            return [
                ChordEvent(
                    time=c.time,
                    chord=c.chord,
                    confidence=c.confidence,
                )
                for c in chords
            ]
        except ImportError:
            print("madmom がインストールされていません。librosa 方式を使用します。")
            return self._detect_with_librosa(y)
    
    def _infer_chord_type(
        self, 
        chroma: np.ndarray, 
        strongest_notes: np.ndarray
    ) -> Tuple[str, float]:
        """
        chroma feature から和音タイプを推定する（簡易ルールベース）
        
        Parameters
        ----------
        chroma : np.ndarray
            12 次元の chroma ベクトル
        strongest_notes : np.ndarray
            最も強い音符のインデックス（0～11）
        
        Returns
        -------
        Tuple[str, float]
            (和音ラベル，信頼度)
        """
        # トニック候補を決定
        tonic_idx = strongest_notes[0]
        
        # メジャー/マイナーの判定
        third_idx = (tonic_idx + 4) % 12  # メジャーサード
        fifth_idx = (tonic_idx + 7) % 12   # パーフェクトフィフス
        
        tonic_strength = chroma[tonic_idx]
        third_strength = chroma[third_idx]
        fifth_strength = chroma[fifth_idx]
        
        # メジャー/マイナーの判定
        if third_strength > fifth_strength:
            mode = 'major'
        else:
            mode = 'minor'
        
        # 和音ラベルを生成
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        chord_label = f"{note_names[tonic_idx]}{mode}"
        
        # 信頼度を計算（簡易）
        confidence = (tonic_strength + third_strength + fifth_strength) / 3
        
        return chord_label, float(confidence)
```

---

## 3. キー推定モジュール（[`key_estimator.py`](src/allin1/key_detection/key_estimator.py)）

### 3.1 クラス設計

```python
@dataclass
class KeyResult:
    """キー推定結果"""
    key: str           # キー（例：'C major', 'A minor'）
    confidence: float  # 推定信頼度（0.0～1.0）
    tonic: str         # トニック音（例：'C'）
    mode: str          # モード（'major' または 'minor'）


class KeyEstimator:
    """キー推定モジュール"""
    
    def __init__(self, method: str = 'music21'):
        """
        Parameters
        ----------
        method : str
            推定手法（'music21' または 'statistical'）
        """
        self.method = method
    
    def estimate_key(self, chords: List[ChordEvent]) -> KeyResult:
        """
        和音シーケンスからキーを推定する
        
        Parameters
        ----------
        chords : List[ChordEvent]
            和音イベントのリスト
        
        Returns
        -------
        KeyResult
            キー推定結果
        """
        if self.method == 'music21':
            return self._estimate_with_music21(chords)
        else:
            return self._estimate_statistical(chords)
    
    def _estimate_with_music21(self, chords: List[ChordEvent]) -> KeyResult:
        """music21 を使用してキーを推定"""
        try:
            from music21 import chord as m21_chord, key
            
            # 和音シーケンスを music21 の Chord オブジェクトに変換
            m21_chords = []
            for ce in chords:
                try:
                    c = m21_chord.Chord(ce.chord)
                    m21_chords.append(c)
                except Exception:
                    continue
            
            if not m21_chords:
                return KeyResult(key='unknown', confidence=0.0, tonic='', mode='')
            
            # キー推定
            k = key.parse(m21_chords)
            
            return KeyResult(
                key=f"{k.tonic.name} {k.mode}",
                confidence=float(k.confidence),
                tonic=k.tonic.name,
                mode=k.mode,
            )
        except ImportError:
            print("music21 がインストールされていません。統計的方式を使用します。")
            return self._estimate_statistical(chords)
    
    def _estimate_statistical(self, chords: List[ChordEvent]) -> KeyResult:
        """
        統計的方式でキーを推定（Krumhansl-Schmiedler プロファイル使用）
        
        各キーに対する和音の適合度をスコアリングし、最も高いキーを選択する。
        """
        # Krumhansl-Schmiedler プロファイル（メジャー/マイナー）
        major_profile = [
            6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
            2.19, 3.66, 2.29, 1.19, 2.39, 1.43
        ]
        minor_profile = [
            6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
            2.19, 3.66, 2.29, 1.19, 2.39, 1.43
        ]
        
        # 和音の頻度をカウント
        chord_counts = Counter()
        for ce in chords:
            # トニック音を抽出（簡易）
            tonic = ce.chord[0]
            chord_counts[tonic] += ce.confidence
        
        # 各キーに対する適合度を計算
        key_scores = {}
        note_to_idx = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5,
                       'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
        
        for tonic in note_to_idx:
            score = 0.0
            for chord, count in chord_counts.items():
                idx = note_to_idx.get(chord, 0)
                profile_score = major_profile[idx] if chord.endswith('major') else minor_profile[idx]
                score += count * profile_score
            key_scores[tonic] = score
        
        # 最も高いスコアのキーを選択
        best_tonic = max(key_scores, key=key_scores.get)
        
        return KeyResult(
            key=f"{best_tonic} major",
            confidence=key_scores[best_tonic] / sum(key_scores.values()),
            tonic=best_tonic,
            mode='major',
        )
```

---

## 4. 小節アライメントモジュール（[`measure_aligner.py`](src/allin1/key_detection/measure_aligner.py)）

### 4.1 クラス設計

```python
@dataclass
class Measure:
    """小節情報"""
    start: float           # 開始時刻（秒）
    end: float             # 終了時刻（秒）
    chords: List[ChordEvent] = field(default_factory=list)  # 含まれる和音イベント
    key_result: Optional[KeyResult] = None  # キー推定結果（後で設定）


class MeasureAligner:
    """小節アライメント処理"""
    
    def __init__(self, downbeats: List[float], bpm: Optional[int] = None):
        """
        Parameters
        ----------
        downbeats : List[float]
            ダウンビート時刻のリスト（秒）
        bpm : Optional[int]
            BPM 情報（オプション、ダウンビートから自動計算される）
        """
        self.downbeats = sorted(downbeats)
        self.bpm = bpm or self._estimate_bpm_from_downbeats()
    
    def _estimate_bpm_from_downbeats(self) -> int:
        """ダウンビート間隔から BPM を推定"""
        if len(self.downbeats) < 2:
            return 120  # デフォルト値
        
        intervals = np.diff(self.downbeats)
        avg_interval = np.median(intervals)
        bpm = int(round(60.0 / avg_interval))
        
        # 妥当な範囲にクランプ
        return max(40, min(240, bpm))
    
    def align_chords_to_measures(self, chords: List[ChordEvent]) -> List[Measure]:
        """
        和音イベントを小節にアライメントする
        
        Parameters
        ----------
        chords : List[ChordEvent]
            和音イベントのリスト
        
        Returns
        -------
        List[Measure]
            アライメントされた小節のリスト
        """
        measures = []
        
        for i in range(len(self.downbeats) - 1):
            measure_start = self.downbeats[i]
            measure_end = self.downbeats[i + 1]
            
            # この小節に含まれる和音をフィルタリング
            measure_chords = [
                c for c in chords 
                if measure_start <= c.time < measure_end
            ]
            
            measures.append(Measure(
                start=measure_start,
                end=measure_end,
                chords=measure_chords,
            ))
        
        # 最後のダウンビート以降の和音を処理
        if len(self.downbeats) > 0:
            last_downbeat = self.downbeats[-1]
            remaining_chords = [c for c in chords if c.time >= last_downbeat]
            
            if remaining_chords:
                measures.append(Measure(
                    start=last_downbeat,
                    end=max(c.time + 0.5 for c in remaining_chords),  # 推定終了時刻
                    chords=remaining_chords,
                ))
        
        return measures
    
    def get_measures_in_segment(
        self, 
        segment: Segment,
        all_measures: List[Measure]
    ) -> List[Measure]:
        """
        セグメントに含まれる小節を抽出する
        
        Parameters
        ----------
        segment : Segment
            セグメント情報
        all_measures : List[Measure]
            全小節のリスト
        
        Returns
        -------
        List[Measure]
            セグメントに含まれる小節のリスト
        """
        return [
            m for m in all_measures
            if m.start >= segment.start and m.end <= segment.end
        ]
```

---

## 5. 集約モジュール（[`aggregator.py`](src/allin1/key_detection/aggregator.py)）

### 5.1 クラス設計

```python
@dataclass
class SegmentKeyInfo(Segment):
    """セグメントとキー情報の組み合わせ"""
    key: str           # 代表キー
    confidence: float  # 推定信頼度
    measure_count: int = 0  # 含まれる小節数


class KeyAggregator:
    """セグメント集約処理"""
    
    def aggregate_by_segment(
        self, 
        measures: List[Measure], 
        segments: List['Segment']
    ) -> List[SegmentKeyInfo]:
        """
        セグメントごとにキーを集約する
        
        Parameters
        ----------
        measures : List[Measure]
            小節のリスト
        segments : List[Segment]
            セグメントのリスト
        
        Returns
        -------
        List[SegmentKeyInfo]
            セグメントごとのキー情報
        """
        results = []
        
        for segment in segments:
            # セグメントに含まれる小節を抽出
            seg_measures = self._get_measures_in_segment(segment, measures)
            
            if not seg_measures:
                # 小節がない場合はデフォルト値を設定
                results.append(SegmentKeyInfo(
                    start=segment.start,
                    end=segment.end,
                    label=segment.label,
                    key='unknown',
                    confidence=0.0,
                    measure_count=0,
                ))
                continue
            
            # 小節のキーを集約
            keys = [m.key_result for m in seg_measures if m.key_result is not None]
            
            if not keys:
                results.append(SegmentKeyInfo(
                    start=segment.start,
                    end=segment.end,
                    label=segment.label,
                    key='unknown',
                    confidence=0.0,
                    measure_count=len(seg_measures),
                ))
                continue
            
            # 最も信頼度の高いキーを選択
            best_key = max(keys, key=lambda k: k.confidence)
            
            results.append(SegmentKeyInfo(
                start=segment.start,
                end=segment.end,
                label=segment.label,
                key=best_key.key,
                confidence=best_key.confidence,
                measure_count=len(seg_measures),
            ))
        
        return results
    
    def _get_measures_in_segment(
        self, 
        segment: 'Segment',
        measures: List[Measure]
    ) -> List[Measure]:
        """セグメントに含まれる小節を抽出する"""
        return [
            m for m in measures
            if m.start >= segment.start and m.end <= segment.end
        ]
```

---

## 6. メインエントリポイント（[`__init__.py`](src/allin1/key_detection/__init__.py)）

### 6.1 パッケージ設計

```python
from .mixer import AudioMixer, AudioLoadError
from .chord_detector import ChordDetector, ChordEvent, ChordDetectionError
from .key_estimator import KeyEstimator, KeyResult, KeyEstimationError
from .measure_aligner import MeasureAligner, Measure
from .aggregator import KeyAggregator, SegmentKeyInfo

__all__ = [
    'AudioMixer',
    'AudioLoadError',
    'ChordDetector', 
    'ChordEvent',
    'ChordDetectionError',
    'KeyEstimator',
    'KeyResult',
    'KeyEstimationError',
    'MeasureAligner',
    'Measure',
    'KeyAggregator',
    'SegmentKeyInfo',
]


def detect_key_per_segment(
    track_path: Path,
    analysis_result: 'AnalysisResult',
    demix_dir: Path = './demix',
) -> List[SegmentKeyInfo]:
    """
    高レベル API：セグメントごとのキー検出
    
    Parameters
    ----------
    track_path : Path
        音声ファイルのパス
    analysis_result : AnalysisResult
        既存の分析結果（ダウンビート，セグメント情報を含む）
    demix_dir : Path
        Demucs 出力ディレクトリ
    
    Returns
    -------
    List[SegmentKeyInfo]
        セグメントごとのキー情報
    """
    # 1. ミキサーで音源を準備
    mixer = AudioMixer(demix_dir=demix_dir)
    y, sr = mixer.mix_without_vocals_drums(track_path)
    
    # 2. 和音検出
    chord_detector = ChordDetector(sr=sr)
    chords = chord_detector.detect_chords(y)
    
    # 3. 小節アライメント
    aligner = MeasureAligner(
        downbeats=analysis_result.downbeats,
        bpm=analysis_result.bpm,
    )
    measures = aligner.align_chords_to_measures(chords)
    
    # 4. キー推定（各小節）
    estimator = KeyEstimator()
    for measure in measures:
        if measure.chords:
            measure.key_result = estimator.estimate_key(measure.chords)
    
    # 5. セグメント集約
    aggregator = KeyAggregator()
    key_info = aggregator.aggregate_by_segment(
        measures=measures,
        segments=analysis_result.segments,
    )
    
    return key_info
```

---

## 7. エラーハンドリング詳細

### 7.1 例外クラス

```python
class KeyDetectionError(Exception):
    """キー検出モジュールの基底例外"""
    pass


class AudioLoadError(KeyDetectionError):
    """音源読み込みエラー"""
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message
    
    def __str__(self):
        return f"AudioLoadError: {self.message}"


class ChordDetectionError(KeyDetectionError):
    """和音検出エラー"""
    pass


class KeyEstimationError(KeyDetectionError):
    """キー推定エラー"""
    pass
```

### 7.2 エラー処理方針

| シナリオ | エラータイプ | 対処法 |
|---------|-------------|--------|
| Demucs 音源なし | `AudioLoadError` | 既存の `analyze()` で再実行を促すメッセージを表示 |
| 和音検出失敗（空） | `ChordDetectionError` | デフォルトキー 'unknown' を設定し、警告を出力 |
| キー推定失敗 | `KeyEstimationError` | 前セグメントのキーを継承 |

---

## 8. テストケース設計

### 8.1 ユニットテスト

```python
# tests/test_key_detection.py

import pytest
from src.allin1.key_detection import (
    AudioMixer, ChordDetector, KeyEstimator, MeasureAligner, KeyAggregator,
)


class TestAudioMixer:
    def test_mix_without_vocals_drums(self, tmp_path):
        # Demucs 音源を仮想的に作成
        demix_dir = tmp_path / 'htdemucs' / 'test_track'
        demix_dir.mkdir(parents=True)
        
        (demix_dir / 'bass.wav').touch()
        (demix_dir / 'other.wav').touch()
        
        mixer = AudioMixer(demix_dir=tmp_path / 'htdemucs')
        # 実装に応じてテストを記述


class TestChordDetector:
    def test_detect_chords(self):
        detector = ChordDetector(sr=44100)
        y = np.zeros(44100 * 5)  # 5 秒の無音
        
        chords = detector.detect_chords(y)
        assert isinstance(chords, list)


class TestKeyEstimator:
    def test_estimate_key(self):
        estimator = KeyEstimator()
        chords = [
            ChordEvent(time=0.0, chord='C', confidence=1.0),
            ChordEvent(time=1.0, chord='Em', confidence=0.9),
            ChordEvent(time=2.0, chord='Am', confidence=0.8),
        ]
        
        result = estimator.estimate_key(chords)
        assert result.key is not None


class TestMeasureAligner:
    def test_align_chords_to_measures(self):
        downbeats = [0.0, 1.875, 3.750]  # 120 BPM の場合
        aligner = MeasureAligner(downbeats=downbeats)
        
        chords = [
            ChordEvent(time=0.5, chord='C', confidence=1.0),
            ChordEvent(time=2.0, chord='Em', confidence=0.9),
        ]
        
        measures = aligner.align_chords_to_measures(chords)
        assert len(measures) == 2


class TestKeyAggregator:
    def test_aggregate_by_segment(self):
        aggregator = KeyAggregator()
        
        measures = [
            Measure(start=0.0, end=1.875, chords=[], key_result=KeyResult(key='C major', confidence=0.9)),
            Measure(start=1.875, end=3.750, chords=[], key_result=KeyResult(key='C major', confidence=0.8)),
        ]
        
        segments = [Segment(start=0.0, end=3.750, label='verse')]
        
        result = aggregator.aggregate_by_segment(measures, segments)
        assert len(result) == 1
        assert result[0].key == 'C major'
```

---

## 9. パフォーマンス最適化

### 9.1 キャッシュ戦略

| キャッシュ対象 | キー | 保存形式 | 有効期間 |
|---------------|------|---------|---------|
| ミックス音源 | `{track_name}_mixed.wav` | WAV | 永続 |
| 和音シーケンス | `{track_name}_chords.json` | JSON | 永続 |
| キー推定結果 | `{track_name}_key.json` | JSON | 永続 |

### 9.2 並列処理

```python
from multiprocessing import Pool

def process_track(args):
    track_path, analysis_result = args
    return detect_key_per_segment(track_path, analysis_result)


def batch_detect_key(
    tracks: List[Tuple[Path, 'AnalysisResult']],
    num_workers: int = 4,
):
    """複数トラックを並列処理"""
    with Pool(num_workers) as pool:
        results = list(pool.map(process_track, tracks))
    return results
```

---

## 10. 参照ドキュメント

- [`01_requirements.md`](docs/01_requirements.md) - 要件定義書
- [`02_system_design.md`](docs/02_system_design.md) - システム設計書
- [`AnalysisResult`](src/allin1/typings.py:32) - 既存の分析結果データ構造
