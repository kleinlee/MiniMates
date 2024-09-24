#include "player.h"

Player::Player()
{
    m_player = new QMediaPlayer(this);
    m_audioOutput = new QAudioOutput(this);
    m_audioOutput->setVolume(50);
    m_player->setAudioOutput(m_audioOutput);

    QObject::connect(m_player, &QMediaPlayer::mediaStatusChanged, [&](QMediaPlayer::MediaStatus status) {
        if (status == QMediaPlayer::EndOfMedia) {
            emit finished();
        }
    });
}

void Player::play(QByteArray& audioData)
{
    qDebug() << "play!!" << audioData.size();
    m_player->setSource(QUrl());
    m_audioBuffer.close();
    m_audioBuffer.setBuffer(&audioData);
    m_player->setSourceDevice(&m_audioBuffer, QUrl("audio.mp3"));
    m_player->play();
}

void Player::abort()
{
    m_player->stop();
}
