**初始化音频接口**

- 使用 `pyaudio.PyAudio()` 创建音频接口 `p`，用于管理麦克风和扬声器的音频流。

**配置麦克风和扬声器流**

- 分别为麦克风输入和扬声器输出配置 `mic_stream` 和 `speaker_stream`。
- `mic_stream` 使用 `mic_callback` 函数捕获音频帧，将每个音频帧存储到队列 `mic_queue` 中，以便后续通过 WebSocket 发送。
- `speaker_stream` 使用 `speaker_callback` 从 `audio_buffer` 中获取音频数据播放，负责处理播放时间控制和缓冲管理。

**启动音频流**

- 调用 `mic_stream.start_stream()` 和 `speaker_stream.start_stream()` 启动音频捕获和播放。

**建立 WebSocket 连接**

- 通过 

  ```
  connect_to_openai()
  ```

   函数与 OpenAI 的 API 建立 WebSocket 连接以发送和接收实时音频数据。此函数启动两个线程：

  - **发送麦克风音频**：`send_mic_audio_to_websocket` 从 `mic_queue` 中提取音频数据，将其编码为 base64 格式，并通过 WebSocket 发送 `input_audio_buffer.append` 格式的消息。
  - **接收音频和响应**：`receive_audio_from_websocket` 监听来自 OpenAI WebSocket 的消息，并根据消息的 `type` 进行处理。

**接收 WebSocket 事件并处理消息类型**

- 在 `receive_audio_from_websocket` 中，不同类型的 WebSocket 消息处理如下：
  - **`session.created`**：表示成功建立会话。响应该事件，调用 `send_fc_session_update()` 配置会话参数，如转录和语音检测。
  - **`input_audio_buffer.speech_started`**：指示 AI 检测到输入音频中的语音开始。这将触发以下操作：
    - `clear_audio_buffer()` 清除 `audio_buffer` 中现有的数据。
    - `stop_audio_playback()` 停止扬声器流，确保只播放新接收的音频数据。
  - **`response.audio.delta`**：包含音频数据的 `delta` 字段，以 base64 格式编码。该音频数据将被解码后添加到 `audio_buffer` 中，并由 `speaker_callback` 进行实时播放。
  - **`response.audio.done`**：标记 AI 音频响应的结束。此消息指示不会再接收当前响应的音频数据。
  - **`response.function_call_arguments.done`**：表明 AI 请求调用一个函数。`handle_function_call` 函数解码请求参数，执行指定操作（例如 `get_weather` 请求），并使用 `send_function_call_result()` 将结果返回给 AI。

**发送会话更新**

- `send_fc_session_update()` 配置特定的会话参数，例如 AI 响应的语气、速度、语言和音频格式。此消息确保 WebSocket 会话以预期的设置和行为进行语音互动。

**保持音频流活跃**

- 主循环持续检查两个音频流的活动状态（`mic_stream` 和 `speaker_stream`），通过 `is_active()` 确保它们在循环中持续运行，每次循环暂停 0.1 秒以节省资源。

**异常处理和中断监控**

- 如果检测到 `KeyboardInterrupt`，则设置 `stop_event`，信号线程安全地终止音频流和 WebSocket 通信。

**关闭音频流并释放资源**

- 在 `finally` 块中关闭音频流和 WebSocket 连接。调用 `stop_stream()` 和 `close()` 关闭 `mic_stream` 和 `speaker_stream`，并使用 `p.terminate()` 释放 `PyAudio` 资源。