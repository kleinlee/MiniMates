#ifndef HUMANASSETS_H
#define HUMANASSETS_H
#include <QString>
#include <QTimer>
#include <QObject>
#include <QDateTime>
#include "llmmodel.h"
#include "communicate.h"
#include "player.h"
struct TTSBuffer
{
    QList<QString> sub_sentences;
    QList<QByteArray> mp3_buffer;
        int index_playing = 0;
    // 清空数据的成员函数
    void clear() {
        index_playing = 0;
        sub_sentences.clear();
        mp3_buffer.clear();
    }

};

class HumanAssets : public QObject
{
    Q_OBJECT
private:
    HumanAssets();    //构造函数是私有的，这样就不能在其它地方创建该实例

    static HumanAssets *instance_;  //定义一个唯一指向实例的静态指针，并且是私有的。
public:
    static HumanAssets* GetInstance()   //定义一个公有函数，可以获取这个唯一的实例，并且在需要的时候创建该实例。
    {
        if(instance_ == NULL)  //判断是否第一次调用
            instance_ = new HumanAssets();
        return instance_;
    }

public:
    QString m_text;
    QString m_task_id;
    QList<float> m_blendshape;
    QDateTime m_start_time_ms;

    LLMModel* m_chat_model;
    Communicate m_TTS_client;

    int m_port_num;

    QTimer* m_frame_timer;

    int m_runing_audio_decoding = 0;  // 处理到第几个句子
    bool m_is_audio_decoding = false;

    bool m_is_audio_playing = false;

    TTSBuffer m_tts_buffer;
    Player m_audio_player;

public slots:
    void SlotNewChat(QString question);
    void SlotStopChat();
    void SlotNewAnswer(QString str);

    void updateFrame();

    void abort();
};

#endif // HUMANASSETS_H
