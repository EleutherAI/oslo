import threading

import pika
# from pika.adapters.blocking_connection import BlockingChannel


CHANNEL = None


def master_handshake_callback(channel, method, properties, body):
    print(body)


def slave_handshake_callback(channel, method, properties, body):
    print(body)


def start_consume(*args, **kwargs):
    if (
            "callback" not in kwargs
            or "queue" not in kwargs
            or not callable(kwargs["callback"])
    ):
        return None

    global CHANNEL
    assert CHANNEL is not None

    callback = kwargs["callback"]
    queue = kwargs["queue"]
    consumer_tag = kwargs["consumer_tag"]

    CHANNEL.basic_consume(
        on_message_callback=callback,
        queue=queue,
        consumer_tag=consumer_tag,
    )

    t = threading.Thread(target=CHANNEL.start_consuming)
    t.setDaemon(True)
    t.start()


def start_master():
    global CHANNEL

    if CHANNEL is None:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost')     # TODO;
        )
        CHANNEL = connection.channel()

    CHANNEL.queue_declare(queue='to_master')
    CHANNEL.queue_declare(queue='from_master')


def start_slave():
    global CHANNEL

    if CHANNEL is None:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='localhost')  # TODO;
        )
        CHANNEL = connection.channel()

    CHANNEL.queue_declare(queue='to_slave')
    CHANNEL.queue_declare(queue='from_slave')


def request_master(from_, to_):
    global CHANNEL

    CHANNEL.basic_publish(
        exchange='',
        routing_key='to_master',
        body='Hello World',
    )
