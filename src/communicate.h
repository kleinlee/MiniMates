#ifndef COMMUNICATE_H
#define COMMUNICATE_H

#include <QDir>
#include <QUuid>
#include <QFile>
#include <QFileInfo>
#include <QProcess>
#include <QWebSocket>
#include <QDesktopServices>
#include <QtMultimedia/QMediaPlayer>
#include <QtMultimedia/QAudioOutput>
#include <QBuffer>
#include <QQueue>
#include <QAudioDecoder>
// Class for communicating with the service
class Communicate : public QObject
{
    Q_OBJECT

public:
    Communicate(QObject *parent = nullptr);

    ~Communicate();

    void setText(QString text);

    void setVoice(QString voice);

    void abort();

    QByteArray getAudioBuffer();

public slots:
    void start();

private slots:
    void onConnected();

    void onBinaryMessageReceived(const QByteArray &message);

    void onTextMessageReceived(const QString &message);

    void onDisconnected();

    void sendNextTextPart();

    void onDecoderBufferReady();

signals:
    void finished();

private:
    QString m_text;
    QString m_voice = "zh-CN, XiaoxiaoNeural";
    QString m_rate = "+0%";
    QString m_volume = "+0%";
    QString m_pitch = "+0Hz";
    QWebSocket m_webSocket;
    QByteArray m_audioDataReceived = "";
    bool m_downloadAudio = false;
    qsizetype m_textPartIndex;
    QString m_date;
    bool m_isDuplicated = false;
    QBuffer m_audioBuffer;
    qsizetype m_audioOffset;

    QAudioDecoder *m_decoder;

    static const qsizetype ms_maxMessageSize = 8192 * 16;
    static const qsizetype ms_startupSize = 8192 * 4;

private:
    // Utility functions
    QString connect_id();

    QString date_to_string();

    QString escape(QString data);

    QString remove_incompatible_characters(QString str);

    QString mkssml(QString text, QString voice, QString rate, QString volume, QString pitch);

    QString ssml_headers_plus_data(const QString& requestId, const QString& timestamp, const QString& ssml);

    QPair<QMap<QString, QString>, QString> get_headers_and_data(const QString& message);
};

#endif // COMMUNICATE_H
