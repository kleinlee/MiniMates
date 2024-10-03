#include "humanassets.h"

HumanAssets* HumanAssets::instance_ = NULL;
HumanAssets::HumanAssets()
{
//    instance = NULL;
    m_chat_model = new LLMModel;
//    connect(ui->pushButton_stop, SIGNAL(clicked()), m_chat_model, SLOT(Reset()));

    m_frame_timer = new QTimer(this);
    m_frame_timer->setTimerType(Qt::PreciseTimer);
    connect(m_frame_timer,&QTimer::timeout,this, &HumanAssets::updateFrame);
    m_frame_timer->start(40);

    connect(&m_TTS_client, &Communicate::finished, [&]() {
        m_is_audio_decoding = false;
        m_tts_buffer.mp3_buffer.append(m_TTS_client.getAudioBuffer());
        qDebug() << "mp3_buffer: " << m_tts_buffer.mp3_buffer.size() << m_tts_buffer.mp3_buffer.last().length();
    });

    connect(&m_audio_player, &Player::finished, [&]() {
        m_is_audio_playing = false;
        qDebug() << "mp3_playing: " << m_tts_buffer.index_playing;
    });
}

void HumanAssets::SlotStopChat()
{
    m_chat_model->Reset();
}
void HumanAssets::SlotNewChat(QString question)
{
    m_TTS_client.abort();
    m_audio_player.abort();
    m_tts_buffer.clear();
    m_is_audio_playing = false;

    m_is_audio_decoding = false;
    m_chat_model->Run(question, m_port_num);
    m_runing_audio_decoding = 0;

}

void HumanAssets::SlotNewAnswer(QString str)
{

}

void HumanAssets::updateFrame()
{
//    if (!m_runing_audio_decoding and m_chat_model->m_llm_text.is_endpoint)
    if (!m_is_audio_decoding and m_chat_model->m_llm_text.sub_sentences.size() > m_runing_audio_decoding)
    {
        m_is_audio_decoding = true;
        QString str_concat;
        for (int i = m_runing_audio_decoding; i < m_chat_model->m_llm_text.sub_sentences.size(); i++) {
            str_concat.append(m_chat_model->m_llm_text.sub_sentences.at(i));
        }
        qDebug() << "start tts:" << str_concat;
        m_runing_audio_decoding = m_chat_model->m_llm_text.sub_sentences.size();
        m_TTS_client.setText(str_concat);
//        m_TTS_client.setVoice("zh-CN, XiaoyiNeural");
        // checkDuplicate(text, voice);
        m_TTS_client.start();
    }

    if (!m_is_audio_playing and m_tts_buffer.mp3_buffer.size() > m_tts_buffer.index_playing)
    {
        m_is_audio_playing = true;
        m_audio_player.play(m_tts_buffer.mp3_buffer[m_tts_buffer.index_playing]);
        m_tts_buffer.index_playing += 1;
    }

    // 更新每一帧的pcm和图像
}
void HumanAssets::abort()
{
    m_audio_player.abort();
    m_tts_buffer.clear();
    m_is_audio_playing = false;

    m_is_audio_decoding = false;
    m_runing_audio_decoding = 0;
    m_chat_model->abort();
    m_TTS_client.abort();

}
