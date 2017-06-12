"""
Contains classes and functions for creating, maintaining, and interacting
with a SQLite database containing data from training runs of various models.
"""

from platform import processor
from datetime import datetime

from sqlalchemy import (create_engine, Column, Table, UniqueConstraint 
                        ForeignKey, Integer, DateTime, Text, Boolean, Float)
from sqlalchemy.orm import sessionmaker, relationship, backref
from sqlalchemy.orm.query import Query
from sqlalchemy.ext.declarative import declarative_base
from psutil import virtual_memory
from GPUtil import getGPUs


MODEL_DIR = os.path.join(os.path.join(os.pardir, "models"), "desc2code_task")
DATABASE_URL = "sqlite://" + os.path.join(
    MODEL_DIR, "desc2code_training_database")
Base = declarative_base()


def _create_schema():
    engine = create_engine(DATABASE_URL, echo=False)
    Base.metadata.create_all(engine)


def start_session():
    engine = create_engine(DATABASE_URL, echo=False)
    return sessionmaker(bind=engine)()


class Model(Base):
    __tablename__ = "Model"

    model_id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime)

    # Model architecture
    description = Column(Text)
    input_type = Column(Text)
    output_type = Column(Text)
    encoder_cell = Column(Text)
    decoder_cell = Column(Text)
    layers = Column(Integer)
    bidirectional_encoder = Column(Boolean)
    attention = Column(Boolean)

    # Other hyperparameters
    hidden_dimension = Column(Integer)
    encoder_vocab_size = Column(Integer)
    decoder_vocab_size = Column(Integer)
    encoder_embedding_size = Column(Integer)
    decoder_embedding_size = Column(Integer)
    batch_size = Column(Integer)
    truncated_backprop = Column(Boolean)
    learning_rate = Column(Float)
    optimization_algorithm = Column(Text)


class TrainingRun(Base):
    __tablename__ = "Training_Run"

    run_id = Column(Integer, primary_key=True)
    begin_timestamp = Column(DateTime)
    end_timestamp = Column(DateTime)
    used_gpus = Column(Integer)

    # Computer info
    cpu = Column(Text)
    gpu = Column(Text)
    ram = Column(Integer)

    model_id = Column(Integer, ForeignKey("Model.model_id"))
    model = relationship("Model", backref="training_run")

    dataset_id = Column(Integer, ForeignKey("Dataset_Info.dataset_id"))
    dataset = relationship("DatasetInfo", backref="training_run")


class DatasetInfo(Base):
    __tablename__ = "Dataset_Info"

    dataset_id = Column(Integer, primary_key=True)
    training_fraction = Column(Float)
    validation_fraction = Column(Float)

    # Dataset info
    input_sequences = Column(Integer)
    average_input_length = Column(Float)
    smallest_input_length = Column(Integer)
    largest_input_length = Column(Integer)

    output_sequences = Column(Integer)
    average_output_length = Column(Float)
    smallest_output_length = Column(Integer)
    largest_output_length = Column(Integer)


class ModelEvaluation(Base):
    __tablename__ = "Model_Evaluation"

    evaluation_id = Column(Integer, primary_key=True)
    dataset_type = Column(Text) # Training, validation, or test
    timestamp = Column(DateTime)
    loss = Column(Float)
    batch = Column(Integer)
    epoch = Column(Integer)
    step = Column(Integer)

    run_id = Column(Integer, ForeignKey("Training_Run.run_id"))
    training_run = relationship("TrainingRun", backref="evaluation_steps")


def store_model_info(description, input_type, output_type, encoder_cell,
                     decoder_cell, layers, bidirectional_encoder, attention,
                     hidden_dimension, encoder_vocab_size, decoder_vocab_size,
                     encoder_embedding_size, decoder_embedding_size, batch_size, 
                     truncated_backprop, learning_rate, optimization_algorithm,
                     timestamp=datetime.utcnow()):
    session = start_session()
    model = Model(description=description, 
                  input_type=input_type, 
                  output_type=output_type, 
                  encoder_cell=encoder_cell,
                  decoder_cell=decoder_cell, 
                  layers=layers, 
                  bidirectional_encoder=bidirectional_encoder, 
                  attention=attention, 
                  hidden_dimension=hidden_dimension, 
                  encoder_vocab_size=encoder_vocab_size, 
                  decoder_vocab_size=decoder_vocab_size,
                  encoder_embedding_size=encoder_embedding_size, 
                  decoder_embedding_size=decoder_embedding_size, 
                  batch_size=batch_size, 
                  truncated_backprop=truncated_backprop, 
                  learning_rate=learning_rate, 
                  optimization_algorithm=optimization_algorithm, 
                  timestamp=timestamp)
    session.add(model)
    try:
        session.commit()
    except Exception as e:
        session.rollback()
        raise RuntimeError("Failed to store model!")

    return model.model_id


def store_training_run(model_id, begin_timestamp, input_sequences, 
                       average_input_length, smallest_input_length,
                       largest_input_length, output_sequences, 
                       average_output_length, smallest_output_length,
                       largest_output_length, end_timestamp=datetime.utcnow(), 
                       used_gpus=1, cpu=None, gpu=None, ram=None, 
                       training_fraction=0.8, validation_fraction=0.1):
    if cpu is None:
        cpu = processor()
    if ram is None:
        ram = int(round(virtual_memory().total / 1024**3))
    if gpu is None:
        gpu = getGPUs()[0].name

    session = start_session()
    
    dataset = DatasetInfo(training_fraction=training_fraction,
                          validation_fraction=validation_fraction,
                          input_sequences=input_sequences,
                          average_input_length=average_input_length,
                          smallest_input_length=smallest_input_length,
                          largest_input_length=largest_input_length,
                          output_sequences=output_sequences,
                          average_output_length=average_output_length,
                          smallest_output_length=smallest_output_length,
                          largest_output_length=largest_output_length)

    training_run = TrainingRun(begin_timestamp=begin_timestamp,
                               end_timestamp=end_timestamp,
                               used_gpus=used_gpus,
                               cpu=cpu, gpu=gpu, ram=ram)
    training_run.model_id = model_id
    training_run.dataset = dataset

    session.add(training_run)
    try:
        session.commit()
    except Exception as e:
        session.rollback()
        raise RuntimeError("Failed to store training run!")

    return training_run.run_id


def store_evaluation_track(run_id, dataset_type, evaluation_track):
    """
    Args:
        dataset_type: String, accepts 'training', 'validation', or 'test'.
        evaluation_track: List, contains elements of the form 
            (loss, timestamp, epoch, batch).
    """
    session = start_session()
    for i, (loss, timestamp, epoch, batch) in enumerate(evaluation_track):
        model_evaluation = ModelEvaluation(
            dataset_type=dataset_type, loss=loss, timestamp=timestamp, 
            epoch=epoch, batch=batch, step=i)
        model_evaluation.run_id = run_id
        session.add(model_evaluation)

    try:
        session.commit()
    except Exception as e:
        session.rollback()
        raise RuntimeError("Failed to store evaluation track!")
