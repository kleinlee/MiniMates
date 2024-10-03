// Copyright (C) 2017 The Qt Company Ltd.
// SPDX-License-Identifier: LicenseRef-Qt-Commercial OR BSD-3-Clause

#include "sqlconversationmodel.h"

#include <QDateTime>
#include <QDebug>
#include <QSqlError>
#include <QSqlRecord>
#include <QSqlQuery>
#include <QEventLoop>
static const char *conversationsTableName = "Conversations";

static void createTable()
{
    if (QSqlDatabase::database().tables().contains(conversationsTableName)) {
        // The table already exists; we don't need to do anything.
        return;
    }

    QSqlQuery query;
    if (!query.exec(
        "CREATE TABLE IF NOT EXISTS 'Conversations' ("
//        "'id' INTEGER PRIMARY KEY,"
        "'author' TEXT NOT NULL,"
        "'recipient' TEXT NOT NULL,"
        "'timestamp' TEXT NOT NULL,"
        "'message' TEXT NOT NULL,"
        "FOREIGN KEY('author') REFERENCES Contacts ( name ),"
        "FOREIGN KEY('recipient') REFERENCES Contacts ( name )"
        ")")) {
        qFatal("Failed to query database: %s", qPrintable(query.lastError().text()));
    }

//    query.exec("INSERT INTO Conversations VALUES(1, 'Me', 'huahua', '2016-01-01T11:24:53', 'Hi!')");
    query.exec("INSERT INTO Conversations VALUES('huahua', 'Me', '2016-01-07T14:36:16', '我是Qwen。')");
}

SqlConversationModel::SqlConversationModel(QObject *parent) :
    QSqlTableModel(parent)
{
    m_human_assets = HumanAssets::GetInstance();
    createTable();
    setTable(conversationsTableName);
    setSort(2, Qt::DescendingOrder);
    // Ensures that the model is sorted correctly after submitting a new row.
    setEditStrategy(QSqlTableModel::OnManualSubmit);

//    m_timer = new QTimer(this);
//    connect(m_timer, &QTimer::timeout, this, &SqlConversationModel::updateMsg);

    connect(this->m_human_assets->m_chat_model, SIGNAL(SignalNewAnswer()), this, SLOT(SlotNewAnswer()));

}

void SqlConversationModel::SlotNewAnswer()
{
    m_chatEnded = m_human_assets->m_chat_model->m_llm_text.is_endpoint;
    qDebug() << "SlotNewAnswer end" << m_chatEnded;

    if (m_chatEnded)
    {
        emit SignalChatFinishedChanged();
        return;
    }

    m_timestamp = QDateTime::currentDateTime().toString(Qt::ISODateWithMs);
    QString str = m_human_assets->m_chat_model->m_llm_text.text;
//    qDebug() << "SlotNewAnswer" << str << rowCount() << m_timestamp;
    QSqlRecord newRecord = record();
//    newRecord.setValue("id", m_chat_round + 1);
    newRecord.setValue("author", "huahua");
    newRecord.setValue("recipient", "Me");
    newRecord.setValue("timestamp", m_timestamp);
    newRecord.setValue("message", m_human_assets->m_chat_model->m_llm_text.text);
    if (sub_text_index == 0)
    {
        if (!insertRecord(rowCount(), newRecord)) {
            qWarning() << "Failed to send message:" << lastError().text();
            return;
        }
    }
    else
    {
        if (!setRecord(0, newRecord)) {
            qWarning() << "Failed to send message:" << lastError().text();
            return;
        }
    }

//    qDebug() << "sssss" << rowCount() << str ;
    submitAll();
    sub_text_index ++;
}

QString SqlConversationModel::recipient() const
{
    return m_recipient;
}

void SqlConversationModel::setRecipient(const QString &recipient)
{
    if (recipient == m_recipient)
        return;

    m_recipient = recipient;

    const QString filterString = QString::fromLatin1(
        "(recipient = '%1' AND author = 'Me') OR (recipient = 'Me' AND author='%1')").arg(m_recipient);
    setFilter(filterString);
    select();

    emit recipientChanged();
}

QVariant SqlConversationModel::data(const QModelIndex &index, int role) const
{
    if (role < Qt::UserRole)
        return QSqlTableModel::data(index, role);

    const QSqlRecord sqlRecord = record(index.row());
    return sqlRecord.value(role - Qt::UserRole);
}

QHash<int, QByteArray> SqlConversationModel::roleNames() const
{
    QHash<int, QByteArray> names;
    names[Qt::UserRole + 0] = "author";
    names[Qt::UserRole + 1] = "recipient";
    names[Qt::UserRole + 2] = "timestamp";
    names[Qt::UserRole + 3] = "message";
    return names;
}

void SqlConversationModel::sendMessage(const QString &recipient, const QString &message)
{
    m_chatEnded = false;
    emit SignalChatFinishedChanged();
    m_chat_round = rowCount() + 1;
    qDebug() << "sendMessage" << recipient << message << rowCount();
    m_timestamp = QDateTime::currentDateTime().toString(Qt::ISODateWithMs);
    m_recipient = recipient;
    m_message = message;

    QSqlRecord newRecord = record();
//    newRecord.setValue("id", m_chat_round);
    newRecord.setValue("author", "Me");
    newRecord.setValue("recipient", recipient);
    newRecord.setValue("timestamp", m_timestamp);
    newRecord.setValue("message", m_message);
    if (!insertRecord(rowCount(), newRecord)) {
        qWarning() << "Failed to send message:" << lastError().text();
        return;
    }
    qDebug() << "TTTTTTTT" << recipient << message;
    submitAll();
    m_human_assets->SlotNewChat(m_message);
    sub_text_index = 0;
}

void SqlConversationModel::removeAllMessage()
{
    qDebug() << "removeAllMessage" << m_chatEnded;
    if (m_chatEnded)
    {
        removeRows(0, rowCount() - 1);
    }
    submitAll();
}
void SqlConversationModel::abort()
{
    m_human_assets->abort();
    m_chatEnded = true;
    emit SignalChatFinishedChanged();
}
