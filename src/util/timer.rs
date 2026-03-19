use std::error::Error;
use std::fmt::{Display, Formatter};
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct FrameTimer {
    started_at: Instant,
}

impl FrameTimer {
    pub fn start() -> Self {
        Self {
            started_at: Instant::now(),
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.started_at.elapsed()
    }

    pub fn restart(&mut self) -> Duration {
        let elapsed = self.elapsed();
        self.started_at = Instant::now();
        elapsed
    }
}

#[derive(Debug, Clone)]
pub struct FixedStepClock {
    step: Duration,
    max_frame_time: Duration,
    accumulator: Duration,
}

impl FixedStepClock {
    pub fn new(step: Duration, max_frame_time: Duration) -> Result<Self, TimerError> {
        if step.is_zero() {
            return Err(TimerError::ZeroStep);
        }

        if max_frame_time.is_zero() {
            return Err(TimerError::ZeroMaxFrameTime);
        }

        Ok(Self {
            step,
            max_frame_time,
            accumulator: Duration::ZERO,
        })
    }

    pub fn step(&self) -> Duration {
        self.step
    }

    pub fn accumulator(&self) -> Duration {
        self.accumulator
    }

    pub fn set_step(&mut self, step: Duration) -> Result<(), TimerError> {
        if step.is_zero() {
            return Err(TimerError::ZeroStep);
        }

        self.step = step;
        Ok(())
    }

    pub fn clear_accumulator(&mut self) {
        self.accumulator = Duration::ZERO;
    }

    pub fn accumulate_and_consume(&mut self, real_dt: Duration) -> u32 {
        let clamped_dt = real_dt.min(self.max_frame_time);
        self.accumulator += clamped_dt;

        let mut steps = 0;
        while self.accumulator >= self.step {
            self.accumulator -= self.step;
            steps += 1;
        }

        steps
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TimerError {
    ZeroStep,
    ZeroMaxFrameTime,
}

impl Display for TimerError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroStep => write!(f, "fixed-step duration must be greater than zero"),
            Self::ZeroMaxFrameTime => write!(f, "maximum frame time must be greater than zero"),
        }
    }
}

impl Error for TimerError {}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::FixedStepClock;

    #[test]
    fn fixed_step_clock_consumes_multiple_steps() {
        let mut clock =
            FixedStepClock::new(Duration::from_millis(10), Duration::from_secs_f32(0.25))
                .expect("clock should construct");

        let steps = clock.accumulate_and_consume(Duration::from_millis(55));

        assert_eq!(steps, 5);
        assert!(clock.accumulator().as_secs_f32() > 0.0);
    }

    #[test]
    fn fixed_step_clock_clamps_large_frames() {
        let mut clock =
            FixedStepClock::new(Duration::from_millis(10), Duration::from_secs_f32(0.1))
                .expect("clock should construct");

        let steps = clock.accumulate_and_consume(Duration::from_secs_f32(1.0));

        assert_eq!(steps, 10);
    }
}
