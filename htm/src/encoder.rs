use crate::CpuSDR;
use std::ops::{RangeBounds, Range};
use chrono::{DateTime, Utc, TimeZone, Datelike, Timelike, NaiveTime};

pub trait EncoderTarget{
    fn push(&mut self, neuron_index:u32);
}

pub trait Encoder<T>{
    fn encode(&self, sdr:&mut impl EncoderTarget, scalar:T);
}


pub struct FloatEncoder{
    neuron_range_begin:u32,//inclusive
    neuron_range_end:u32,//exclusive
    scalar_range_begin:f32,
    scalar_range_end:f32,
    buckets_per_value:f32,
    sdr_cardinality: u32,
}
impl Encoder<f32> for FloatEncoder{
    fn encode(&self, sdr:&mut impl EncoderTarget, scalar:f32) {
        let scalar = scalar.clamp(self.scalar_range_begin,self.scalar_range_end);
        let diff = scalar - self.scalar_range_begin;
        let offset =  diff*self.buckets_per_value;
        let begin = self.neuron_range_begin + offset as u32;
        let end = begin+self.sdr_cardinality;
        assert!(self.neuron_range_begin<=begin);
        assert!(end <= self.neuron_range_end,"{}, {} + {} = {} <= {}",offset,begin,self.sdr_cardinality,begin+self.sdr_cardinality,self.neuron_range_end);
        for neuron_idx in begin..end{
            sdr.push(neuron_idx)
        }
    }
}

pub struct CategoricalEncoder{
    neuron_range_begin:u32,//inclusive
    num_of_categories:u32,
    sdr_cardinality: u32,
}
impl Encoder<u32> for CategoricalEncoder{
    fn encode(&self, sdr:&mut impl EncoderTarget, scalar:u32) {
        let scalar = scalar%self.num_of_categories;
        let begin = self.neuron_range_begin + scalar * self.sdr_cardinality;
        let end = begin+self.sdr_cardinality;
        for neuron_idx in begin..end{
            sdr.push(neuron_idx)
        }
    }
}


pub struct IntegerEncoder{
    neuron_range_begin:u32,//inclusive
    neuron_range_end:u32,//exclusive
    scalar_range_begin:u32,//inclusive
    scalar_range_end:u32,//exclusive
    buckets_per_value:f32,
    sdr_cardinality: u32,
}
impl Encoder<u32> for IntegerEncoder{
    fn encode(&self, sdr:&mut impl EncoderTarget, scalar:u32) {
        let scalar = scalar.clamp(self.scalar_range_begin,self.scalar_range_end-1);
        let diff = scalar - self.scalar_range_begin;
        let offset =  diff as f32*self.buckets_per_value;
        let begin = self.neuron_range_begin + offset as u32;
        let end = begin+self.sdr_cardinality;
        assert!(self.neuron_range_begin<=begin);
        assert!(end <= self.neuron_range_end,"{}, {} + {} = {} <= {}",offset,begin,self.sdr_cardinality,begin+self.sdr_cardinality,self.neuron_range_end);
        for neuron_idx in begin..end{
            sdr.push(neuron_idx)
        }
    }
}


pub struct CircularIntegerEncoder{
    neuron_range_begin:u32,//inclusive
    neuron_range_end:u32,//exclusive
    scalar_range_begin:u32,//inclusive
    scalar_range_end:u32,//exclusive
    buckets_per_value:f32,
    sdr_cardinality: u32,
}
impl Encoder<u32> for CircularIntegerEncoder{
    fn encode(&self, sdr:&mut impl EncoderTarget, scalar:u32) {
        let possible_values = self.scalar_range_end - self.scalar_range_begin;
        let diff = (scalar - self.scalar_range_begin) % possible_values;
        let offset =  diff as f32*self.buckets_per_value;
        let begin = self.neuron_range_begin + offset as u32;
        let end = begin+self.sdr_cardinality;
        assert!(self.neuron_range_begin<=begin);
        assert!(begin<self.neuron_range_end);
        for neuron_idx in begin..end{
            sdr.push(neuron_idx % self.neuron_range_end)
        }
    }
}

pub struct DayOfWeekEncoder{
    enc:CircularIntegerEncoder
}
impl DayOfWeekEncoder{
    pub fn encode_day_of_week(&self, sdr:&mut impl EncoderTarget, day_of_week:u32){
        self.enc.encode(sdr,day_of_week);
    }
}
impl <T:TimeZone> Encoder<&DateTime<T>> for DayOfWeekEncoder{
    fn encode(&self, sdr:&mut impl EncoderTarget, scalar:&DateTime<T>) {
        self.encode_day_of_week(sdr,scalar.weekday().num_days_from_monday());
    }
}

pub struct DayOfMonthEncoder{
    enc:CircularIntegerEncoder
}
impl DayOfMonthEncoder{
    pub fn encode_day_of_month(&self, sdr:&mut impl EncoderTarget, day_of_month:u32){
        self.enc.encode(sdr,day_of_month);
    }
}
impl <T:TimeZone> Encoder<&DateTime<T>> for DayOfMonthEncoder{
    fn encode(&self, sdr:&mut impl EncoderTarget, scalar:&DateTime<T>) {
        self.encode_day_of_month(sdr,scalar.day0());
    }
}

pub struct BoolEncoder{
    enc:IntegerEncoder
}

impl Encoder<bool> for BoolEncoder{
    fn encode(&self, sdr:&mut impl EncoderTarget, scalar:bool) {
        self.enc.encode(sdr,scalar as u32);
    }
}

pub struct IsWeekendEncoder{
    enc:BoolEncoder
}

impl IsWeekendEncoder{
    pub fn encode_is_weekend(&self, sdr:&mut impl EncoderTarget, is_weekend:bool){
        self.enc.encode(sdr,is_weekend);
    }
}
impl <T:TimeZone> Encoder<&DateTime<T>> for IsWeekendEncoder{
    fn encode(&self, sdr:&mut impl EncoderTarget, scalar:&DateTime<T>) {
        self.encode_is_weekend(sdr,scalar.weekday().number_from_monday()>=6);
    }
}

pub struct TimeOfDayEncoder{
    enc:CircularIntegerEncoder
}
impl TimeOfDayEncoder{
    pub fn encode_time_of_day(&self, sdr:&mut impl EncoderTarget, num_seconds_from_midnight:u32){
        self.enc.encode(sdr,num_seconds_from_midnight);
    }
}
impl <T:TimeZone> Encoder<&DateTime<T>> for TimeOfDayEncoder{
    fn encode(&self, sdr:&mut impl EncoderTarget, scalar:&DateTime<T>) {
        self.encode_time_of_day(sdr,scalar.time().num_seconds_from_midnight());
    }
}

pub struct DayOfYearEncoder{
    enc:CircularIntegerEncoder
}
impl DayOfYearEncoder{
    pub fn encode_day_of_year(&self, sdr:&mut impl EncoderTarget, day_of_year:u32){
        self.enc.encode(sdr,day_of_year);
    }
}
impl <T:TimeZone> Encoder<&DateTime<T>> for DayOfYearEncoder{
    fn encode(&self, sdr:&mut impl EncoderTarget, scalar:&DateTime<T>) {
        self.encode_day_of_year(sdr,scalar.date().ordinal0());
    }
}


pub struct EncoderBuilder{
    len:u32,
}



impl EncoderBuilder{
    pub fn new()->Self{
        Self{len:0}
    }
    pub fn input_size(&self)->u32{
        self.len
    }
    /**size=total number of bits (on and off) in an SDR.
    cardinality=number of on bits*/
    pub fn add_integer(&mut self, input_range:Range<u32>, sdr_size:u32, sdr_cardinality:u32)->IntegerEncoder{
        assert!(input_range.start<input_range.end);
        let possible_inputs = input_range.end - input_range.start;
        let buckets = sdr_size - sdr_cardinality + 1;
        let buckets_per_value = ((buckets - 1)  as f32) / (possible_inputs as f32 - 1f32);
        let neuron_range_begin = self.len;
        self.len += sdr_size;
        let neuron_range_end = self.len;
        IntegerEncoder{
            neuron_range_begin,
            neuron_range_end,
            scalar_range_begin: input_range.start,
            scalar_range_end: input_range.end,
            buckets_per_value,
            sdr_cardinality
        }
    }
    pub fn add_categorical(&mut self, number_of_categories:u32, sdr_cardinality:u32)->CategoricalEncoder{
        let neuron_range_begin = self.len;
        self.len += sdr_cardinality * number_of_categories;
        CategoricalEncoder{
            neuron_range_begin,
            num_of_categories: number_of_categories,
            sdr_cardinality
        }
    }
    pub fn add_circular_integer(&mut self, input_range:Range<u32>, sdr_size:u32, sdr_cardinality:u32)->CircularIntegerEncoder{
        assert!(input_range.start<input_range.end);
        let possible_inputs = input_range.end - input_range.start;
        let buckets = sdr_size;
        let buckets_per_value = (buckets  as f32) / (possible_inputs as f32);
        let neuron_range_begin = self.len;
        self.len += sdr_size;
        let neuron_range_end = self.len;
        CircularIntegerEncoder{
            neuron_range_begin,
            neuron_range_end,
            scalar_range_begin: input_range.start,
            scalar_range_end: input_range.end,
            buckets_per_value,
            sdr_cardinality
        }
    }
    pub fn add_float(&mut self, input_range:Range<f32>, sdr_size:u32, sdr_cardinality:u32)->FloatEncoder{
        assert!(input_range.start<input_range.end);
        let possible_inputs = input_range.end - input_range.start;
        let buckets = sdr_size - sdr_cardinality + 1;
        let buckets_per_value = ((buckets - 1)  as f32) / possible_inputs;
        let neuron_range_begin = self.len;
        self.len += sdr_size;
        let neuron_range_end = self.len;
        FloatEncoder{
            neuron_range_begin,
            neuron_range_end,
            scalar_range_begin: input_range.start,
            scalar_range_end: input_range.end,
            buckets_per_value,
            sdr_cardinality
        }
    }
    pub fn add_day_of_week(&mut self, sdr_size:u32, sdr_cardinality:u32)->DayOfWeekEncoder{
        DayOfWeekEncoder{enc:self.add_circular_integer(0..7,sdr_size,sdr_cardinality)}
    }
    pub fn add_day_of_month(&mut self, sdr_size:u32, sdr_cardinality:u32)->DayOfMonthEncoder{
        DayOfMonthEncoder{enc:self.add_circular_integer(0..31,sdr_size,sdr_cardinality)}
    }
    pub fn add_bool(&mut self, sdr_size:u32, sdr_cardinality:u32)->BoolEncoder{
        BoolEncoder{enc:self.add_integer(0..2,sdr_size,sdr_cardinality)}
    }
    pub fn add_is_weekend(&mut self, sdr_size:u32, sdr_cardinality:u32)->IsWeekendEncoder{
        IsWeekendEncoder{enc:self.add_bool(sdr_size,sdr_cardinality)}
    }
    pub fn add_time_of_day(&mut self, sdr_size:u32, sdr_cardinality:u32)->TimeOfDayEncoder{
        TimeOfDayEncoder{enc:self.add_circular_integer(0..86400, sdr_size,sdr_cardinality)}
    }
    pub fn add_day_of_year(&mut self, sdr_size:u32, sdr_cardinality:u32)->DayOfYearEncoder{
        DayOfYearEncoder{enc:self.add_circular_integer(0..366, sdr_size,sdr_cardinality)}
    }
}