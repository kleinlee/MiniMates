#ifndef PLAYER_H
#define PLAYER_H

#include <QObject>
#include <QBuffer>
#include <QtMultimedia/QMediaPlayer>
#include <QtMultimedia/QAudioOutput>
class Player : public QObject
{
    Q_OBJECT
public:
    Player();

    void play(QByteArray& audioData);
    void abort();

    QMediaPlayer* m_player;
    QAudioOutput* m_audioOutput;
    QBuffer m_audioBuffer;
signals:
    void finished();
};

#endif // PLAYER_H
